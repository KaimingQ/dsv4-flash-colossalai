# 02 架构与框架适配（DeepSeek-V4 × ColossalAI × 最新 PyTorch）

> 核心结论：ColossalAI 官方（0.5.0）仅内置 DeepSeek-V3 的 shardformer policy；
> 本项目以最小补丁方式新增 **DeepSeek-V4 专家并行（EP）适配**，
> 补丁存档于 `patches/0001-deepseek_v4-ep-policy-and-lora-adaptation.patch`。

## 1. 模型架构要点（与训练相关的部分）

- `model_type=deepseek_v4`，transformers 5.15.1 原生支持（`DeepseekV4ForCausalLM`）
- 284.33B 参数：43 层，256 路由专家 + 1 共享专家，`moe_intermediate_size=2048`
- MLA 注意力：`q_a_proj / q_b_proj / kv_proj / o_a_proj / o_b_proj`，head_dim=512
- **head_dim=512 超出 FlashAttention 上限（256），仅支持 eager 注意力**（transformers 源码注释明确）
- 3 个 hash 路由 MoE 层（`tid2eid` 静态查表选专家）+ 40 个 topk 路由层
- 专家权重为**融合 3D 参数**（`experts.gate_up_proj [E, 2I, H]`、`down_proj [E, H, I]`），
  不是 ModuleList —— 因此 LoRA 无法注入路由专家，只能注入注意力与共享专家

## 2. 权重：FP8 → BF16 转换（训练前提）

发布权重是推理压缩格式（非专家为 FP8 e4m3 + 块 scale，路由专家为 MXFP4 e2m1 int8 打包），
不能直接训练：
- 训练需要高精度主权重（lr≈2e-5 的更新会被低精度舍入吃掉）；
- HPC-AI 官方博客亦要求「使用 BF16 权重进行微调」。

**源模型**：`$MODEL_ROOT/deepseek-v4-flash-0731`（后训练更强版本，
结构不变，FP4 专家 + FP8 非专家）。
**反量化算法**：全部采用 [FlagOS convert_weight.py](https://github.com/flagos-ai/DeepSeek-V4-FlagOS)
官方实现（已克隆至 `third_party/DeepSeek-V4-FlagOS`）：
FP8 按 128×128 块乘 e8m0 scale；FP4 按 nibble 解包 + E2M1 LUT + 32 元素组 e8m0 scale。

**转换流程**（`scripts/convert_native_to_bf16_v2.py`，全量输出约 530GB 逐专家布局）：
1. 逐分片读取原生 checkpoint，按 FlagOS 算法反量化（专家走 `dequant_fp4_weight`，
   其余走 `weight_dequant`）；
2. 完整手工键名映射（原生 → HF）：MLA 线性层、双层 compressor（含 indexer 子树）、
   hyper-compressor（`hc_*_{base,fn,scale}`）、`gate.bias → e_score_correction_bias` 等；
3. 专家 w1(gate)+w3(up) 拼接为逐专家 `gate_up_proj [2I, H]`，w2 → `down_proj`；
4. 与 HF meta 模型的参数/持久化缓冲区键集合**严格双向校验**（23215 键，零缺失零多余）；
5. 分片落盘 + 清理量化字段；MTP 投机层不参与训练（产物 config `num_nextn_predict_layers=0`）。
下游推理前由 `scripts/merge_ep_shards_to_hf.py` 融合为 3D 布局（`*-fused`），供 HF 直接加载。

**踩坑全记录（重要）**：
1. v1 管线依赖 transformers 5.15 的 `from_pretrained` 自动键映射，实测该映射**不完整**：
   `hc_*_scale` 与 MTP 相关条目缺失 → 这些张量被随机初始化，生成乱码（反复的 `socio socio…`）；
2. 纯 FP8 变体（DeepSeek-V4-Flash-FP8）的 checkpoint 本身缺失全部 43 层 `wo_a.scale`
   （index 有记录、分片中无张量）→ 无法完整反量化，**弃用**；0731 原版完整；
3. 发布 config 的 `moe_intermediate_size`（2048）与模型定义实际专家宽度（4096）不符，
   以模型定义为准；
4. `torch._grouped_mm`（transformers 默认专家实现）在本环境（torch 2.8 + H20）
   触发 device-side assert，推理需设 `config._experts_implementation = "eager"`；
5. v2 初版曾将专家分支在反量化前提前返回，导致专家保持 int8 打包（形状/类型双重异常，
   加载即 size mismatch）——现行版本已修正为**先反量化再收集**，
   并以 2 层小样做端到端验证（形状 [4096,4096] bf16 + 加载零 MISSING + 生成不崩）。

**数值验证**：自产张量与 FlagOS 算法独立手工反量化逐元素对比，误差为 0。

## 3. ColossalAI 适配补丁清单

### 3.1 新增文件
- `colossalai/shardformer/modeling/deepseek_v4.py`：
  - `EpDeepseekV4MoE`：仿 `EpDeepseekV3MoE` 的 all-to-all 专家并行，逐专家精确计算，
    兼容 hash 路由层 `input_ids`，保留 EP/MoE-DP 梯度缩放；
  - `V4Expert / V4ExpertsList / convert_fused_experts_to_modulelist`：**逐专家布局改造**。
- `colossalai/shardformer/policies/deepseek_v4.py`：`DeepseekV4Policy` 等
  （EP 替换 `DecoderLayer.mlp`、FusedRMSNorm；PP 暂不支持，显式断言）。

### 3.1b 逐专家布局：EP 真正省显存的关键（重点设计）

V4 官方实现将 256 个专家融合为两个 3D 参数（`gate_up_proj [E,2I,H]` /
`down_proj [E,H,I]`），而 ColossalAI 的 EP 需要沿专家维切分、且每 rank 只加载/保存自己持有的 1/ep：
融合张量无法部分加载（加载必须全量 568GB 复制到每卡，8 进程 = 4.5TB，不可行）。

方案（与官方 DeepSeek-V3 ModuleList 路径对齐）：
1. `convert_fused_experts_to_modulelist`：在 LazyInitContext 内将融合专家替换为逐专家
   `nn.ModuleList`（新参数经 `torch.empty` 懒初始化，不占显存）；
2. `scripts/split_experts_per_key.py`：一次性将 BF16 checkpoint 的融合专家键拆为
   `experts.N.gate_up_proj/down_proj` 逐专家键（产物 `DeepSeek-V4-Flash-BF16-EP`，等大 530GB，23129 键）；
3. `EpDeepseekV4MoE.setup_process_groups`：仅保留本 rank 专家，其余置 None（`set_tensors_to_none`），
   权重加载只覆盖 1/ep → 专家显存真正切分（每卡专家权重 ≈ 554GB/8 ≈ 69GB）。

### 3.2 修改文件
- `colossalai/shardformer/policies/auto_policy.py`：注册
  `transformers.models.deepseek_v4.modeling_deepseek_v4.DeepseekV4ForCausalLM` → policy。
- `applications/ColossalChat/examples/training_scripts/lora_finetune.py`：
  1. moe plugin 传入 `cpu_offload=args.zero_cpu_offload`；
  2. `deepseek_v4` 强制 `attn_implementation="eager"`；
  3. **预热导入模型模块**：LazyInitContext 会包装 torch 工厂函数，若模型类在懒初始化内首次导入，
     transformers 5.x 的 `torch.LongTensor | None` 签名注解求值会崩溃（`X | None` 对懒函数非法）；
  4. DeepseekV4 LoRA target：`q_a_proj/q_b_proj/kv_proj/o_a_proj/o_b_proj + gate/up/down_proj`（后者仅命中共享专家）；
  5. 新增 `--low_cpu_mem`：`booster.load_model` 流式加载；
  6. `RawConversationDataset` 补 `system_prompt` 参数（coati 新版 API）；
  7. ep>1 时自动调用逐专家布局转换。

### 3.3 版本适配
- 官方 `requirements` 钉 `torch<=2.5.1 / transformers==4.51.3`，与最新环境冲突；
  采用 `pip install -e . --no-deps` + 显式依赖管理绕过（见 `scripts/setup_env.sh`）。
- torch 2.8 下 `--no-deps` 安装与基础导入、autopolicy 解析均通过。

## 4. 单元测试（`scripts/test_ep_moe.py`）

- `get_autopolicy` 正确返回 `DeepseekV4ForCausalLMPolicy`；
- `EpDeepseekV4MoE`（ep_size=1）与原生 `SparseMoeBlock` 输出数值一致；
- 反向传播可回传梯度。

## 5. 遗留风险（如实记录）

- PP（pipeline）forward 未适配（transformers 5.x forward 签名完全不同），当前仅
  `ep + zero + cpu_offload` 路线；8 卡 768GB + 2TB 内存下理论可行，实测见后续文档。
- hash 路由层在打包序列/训练态下 `input_ids` 对齐逻辑按「展平后取前 T 个 token」处理，
  冒烟测试需重点观察该 3 层的 loss 行为。
