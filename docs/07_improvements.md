# 07 改进工作报告：284B × ColossalAI × 8×H20 后训练全流程攻坚记录

> 本报告完整记录从零构建 DeepSeek-V4-Flash（284B）在单机 8×H20（96GB/卡）上
> **权重转换 → 框架适配 → LoRA SFT → DPO/SimPO → ORPO** 全流程中遇到的所有
> 关键困难、定位思路与解决方案，含具体数值、错误现象与代码机制。
> 项目最终形态见 `README.md`，训练结果数据见 `docs/06_report.md`。

## 0. 总览与关键数字

| 阶段 | 核心困难 | 解决方式 | 结果 |
|---|---|---|---|
| 环境 | 依赖版本冲突链（5 处） | 逐一隔离修复并固化进安装脚本 | 一键可复现 |
| 权重转换 | 三轮"生成乱码"，三个独立根因 | FlagOS 算法 + 23215 键双向校验 | 530GB 零缺失零多余 |
| 框架适配 | 官方无 V4 policy；融合专家无法切分 | 自研 EP policy + 逐专家键布局 | 800 行补丁 / 10 文件 |
| 训练 | OOM / NCCL 超时 / 显存死锁（冒烟 10 轮） | 配置调优 + 根因修复 | 三阶段跑通 |
| ORPO | 三层叠加的 nan 数值漏洞 | 诊断探针 + 稳定化重写 | 133 步零 nan |
| SFT loss | EP combine 错位 + 报告聚合偏大 8 倍 | 层级对拍逐层排除 + 手工 CE 复核 | loss 1.07→0.39 |

**模型关键参数**（决定所有技术选型）：284,332,230,231 参数；43 层；
256 路由专家 + 1 共享专家；3 个 hash 路由层（`tid2eid` 静态查表）+ 40 个 topk 层；
MLA 注意力 `head_dim=512`（**超出 FlashAttention 支持上限 256 → 只能 eager 注意力**）；
带 MTP 投机层（训练不参与）。

**硬件**：8 × NVIDIA H20-SXM4（97871 MiB/卡）+ 2.0 TiB 主机内存 +
NFS 共享盘（14TB，写缓存延迟是本项目的持续性工程干扰源）。

---

## 1. 环境与依赖攻坚

**目标栈**：最新 PyTorch（NGC 25.06 镜像，`torch 2.8.0a0+5228986c39` / CUDA 12.9 /
Python 3.12）+ `transformers 5.15.1`（原生 `deepseek_v4` 支持）+ ColossalAI 0.5.0。

**核心冲突**：ColossalAI 官方 requirements 钉死 `torch<=2.5.1`、
`transformers==4.51.3`，与"最新栈"目标直接矛盾。
**决策**：主包与 ColossalChat 均用 `pip install -e ... --no-deps` 安装，
依赖由本项目显式管理并固化进 `scripts/setup_env.sh`（幂等、容器重建后可重入）。

逐个踩过的坑：

| # | 现象 | 根因 | 解决 |
|---|---|---|---|
| 1 | 镜像构建时整个 pip 解析失败 | `pyext` 用了已被 Python 3.12 移除的 `inspect.getargspec`，单包失败拖垮整体 | 从依赖清单移除（仅旧示例使用），Dockerfile 注明原因 |
| 2 | 训练脚本报 `ModuleNotFoundError: fabric` | `--no-deps` 跳过了 colossalai 的运行时依赖 | `setup_env.sh` 显式补齐 `fabric / rpyc / galore_torch` 等 |
| 3 | 训练脚本报 `coati` 缺失 | ColossalChat 是独立包，训练脚本（`lora_finetune.py` 等）依赖它 | `pip install -e applications/ColossalChat --no-deps` |
| 4 | `peft 0.20` 导入即报 `torchao>=0.16 required`（NGC 预装 0.11） | NGC 镜像的 torchao 版本落后 | `pip install --no-deps torchao==0.16.0`；**容器重建后复发过一次**，随即固化进 `setup_env.sh` |
| 5 | FP8 推理内核报 triton 错误 | `kernels` 包下载的 kernel 代码使用旧版 `JITFunction` API，与 triton 3.3 不兼容 | 绕行：训练走 BF16 反量化路线（本就是官方推荐），不依赖推理内核 |

**经验**：依赖冲突逐个安装验证、立即固化进脚本；容器重建后第一件事重跑
`setup_env.sh` 而不是直接训练（第 4 条复发就是教训）。

## 2. 权重转换管线攻坚（最大攻坚点）

### 2.1 为什么必须转换

官方发布为**推理压缩格式**：路由专家为 MXFP4（e2m1，仅 16 档取值，
两个元素打包进一个 int8 字节，每 32 元素组一个 e8m0 scale），
非专家为 FP8 e4m3（128×128 块，每块一个 e8m0 scale）。
不能直接训练的原因：
- `lr≈2e-5` 的梯度更新会被 4bit/8bit 的舍入直接吃掉，必须有高精度主权重；
- 框架的优化器状态、LoRA 注入、梯度反传都要求常规浮点参数；
- HPC-AI 官方博客亦明确"使用 BF16 权重进行微调"。

### 2.2 三轮"生成乱码"的根因排查（每层都独立导致全错）

转出权重后用 2 层小样做生成验证，输出 `socio socio socio…` 乱码。
**关键教训：数值误差验证为 0 ≠ 权重正确**。三轮排查：

| 轮次 | 现象与排查 | 根因 | 解决 |
|---|---|---|---|
| ① | 逐张量反量化误差为 0，但生成乱码。对照模型定义逐键清点，发现加载日志里有被忽略的 MISSING | transformers 5.15 的 `from_pretrained` 自动键映射**不完整**：`hc_attn_scale / hc_ffn_scale`（hyper-compressor）与 MTP 条目缺失 → 这些张量被**随机初始化** | 放弃自动映射，改完整手工映射（v2 管线不再经过 transformers 加载） |
| ② | 逐分片核对时发现 index 与实际文件不一致 | 纯 FP8 变体（DeepSeek-V4-Flash-FP8，274GB）的 checkpoint **本身缺失全部 43 层 `wo_a.scale`**（`model.safetensors.index.json` 里有、分片文件里没有）——官方发布物的隐性残缺 | 弃用该源，改用 0731 完整版（156GB，专家 FP4 + 非专家 FP8） |
| ③ | 合并产物加载报 size mismatch：专家形状是"未解包"尺寸、类型是 int8 | v2 初版**专家分支在反量化之前就 `continue`**，专家保持 int8 打包 | 修正为先 `dequant_fp4_weight` 再收集；2 层小样端到端验证（形状 `[4096,4096]` bf16 + 加载零 MISSING + 生成不崩）后才启动全量 |

### 2.3 最终管线细节（`scripts/convert_native_to_bf16_v2.py`）

1. **反量化算法全部复用 FlagOS 官方工具**（`third_party/DeepSeek-V4-FlagOS/
   convert_weight.py` 的 `weight_dequant` / `dequant_fp4_weight`）：
   FP8 按 128×128 块乘 `2^scale`（e8m0 解码）；
   FP4 按 nibble 解包 + E2M1 查找表 + 每 32 元素组 scale，宽度翻倍；
2. **完整手工键映射**（示例规则）：`.attn. → .self_attn.`、`.ffn. → .mlp.`、
   `.wq_a. → .q_a_proj.`、`.indexer.compressor.wkv. → .compressor.indexer.kv_proj.`、
   `.indexer.ape → .compressor.indexer.position_bias`、
   `.gate.bias → .gate.e_score_correction_bias`（buffer）等；
   双层 compressor 的 indexer 子树规则必须**先于**外层规则应用（顺序敏感）；
3. 专家 w1(gate)+w3(up) 拼接为逐专家 `gate_up_proj [2I, H]`，w2 → `down_proj`；
4. **与 HF meta 模型（`from_config` 于 meta device）的参数 + 持久化缓冲区键集合
   严格双向校验：23215 键，零缺失零多余**（非持久化的 `inv_freq` 类运行时
   buffer 不参与校验，否则误报）；
5. 专家宽度按模型定义修正：config 里 `moe_intermediate_size=2048` 与实际张量
   宽度 4096 不符（又一个"不以 config 为准、以张量为准"的坑）。

产物：逐专家布局 530GB（训练用，88 分片）+ 融合 3D 布局 530GB
（`merge_ep_shards_to_hf.py` 合并，含键集合校验，下游推理用）。

**经验**：转换产物必须**键集合双向断言 + 小规模端到端生成验证**双保险；
官方发布物可能有隐性残缺；分片转换用"先算分片方案再逐片写入释放"控制峰值内存。

## 3. ColossalAI 框架适配攻坚（800 行补丁 / 10 文件）

**背景**：ColossalAI 0.5.0 官方只有 `deepseek_v3.py` policy；V4 在
transformers 5.x 中是全新实现（融合 3D 专家、双层压缩注意力、hash 路由），
直接跑报"无可用 policy"。

### 3.1 `EpDeepseekV4MoE`：all-to-all 专家并行

仿 `EpDeepseekV3MoE` 实现，处理 V4 特有差异：
- **token 路由**：`sort_order = flat_idx.argsort(stable=True)` 按全局专家号排序 →
  `all_to_all_uneven` dispatch 到持有 rank → 局部按本地专家分组计算（`local_order`）→
  专家计算后**先逆置换恢复接收顺序再 combine 回传** → `index_add_(0, sort_order // k, ...)`
  恢复原序并累加同一 token 的 k 个专家输出；
- **梯度缩放**：`DPGradScalerIn/Out`（MoE-DP 组）与 `EPGradScalerIn/Out`（EP 组）
  保证切分后的梯度语义等价；`activate_experts` 必须统计**本 rank 局部专家**
  的激活情况再跨 MoE-DP 组规约（初版误用全局接收计数，已修复）；
- **hash 路由层**：forward 多一个 `input_ids` 参数；打包/不对齐时
  （`ids.numel() != flat.shape[0]`）安全回退为 topk 打分，保证训练不崩；
- **逐专家计算**：`F.linear(x, experts[e].gate_up_proj)` 逐专家精确计算
  （不用广播 matmul——会物化 `[E, N, H]` 中间张量爆显存）。

### 3.2 逐专家键布局改造（284B 装进 768GB 的关键设计）

**困难**：V4 把 256 个专家融合为两个 3D 参数（`[E, 2I, H]` / `[E, H, I]`），
而 EP 要求**每 rank 只加载/保存 1/8 专家**——融合张量无法部分加载
（全量 568GB × 8 进程 = 4.5TB 复制量，不可行）。

**解决**：`V4Expert / V4ExpertsList / convert_fused_experts_to_modulelist`
在懒初始化上下文内把融合专家原地替换为逐专家 ModuleList：
- 新参数用 `torch.empty` 创建（被 `LazyInitContext` 拦截为懒张量），
  **必须传 `dtype=融合张量.dtype`**（否则 `F.linear` dtype mismatch）；
- 权重由逐专家键的 checkpoint 直接加载，物化只覆盖 1/8 专家。

**冒烟阶段最难定位的 bug（全局索引）**：置空非持有专家参数后，
计算路径仍按**局部下标**访问专家 → 命中 None 权重 → 部分 rank 崩溃。
崩溃的 rank 退出集合通信，其余 7 卡在 `all_to_all` 上等待 →
表现为 **NCCL Watchdog 600s 超时**（报错指向通信，根因却在计算）。
**定位手段**：写 8 卡最小复现 `test_ep_moe_dist.py`（随机权重 16 专家小模型，
绕开 15 分钟的全量加载），复现出 `TypeError: None` →
修复为**保持 ModuleList 全局索引、计算时按全局专家号映射**，微复现 5ms/iter 通过。

### 3.3 与最新栈的兼容性修复（逐个击破）

| # | 现象 | 根因 | 解决 |
|---|---|---|---|
| 1 | 懒初始化内 `from_config` 崩溃，栈指向类型注解求值 | `LazyInitContext` 把模块工厂懒化，transformers 5.x 的 `X \| None` 注解在懒环境下**首次导入**即求值崩溃 | 进入懒上下文**之前**预热导入：`_ = AutoModelForCausalLM._model_mapping[type(config)]` |
| 2 | 物化时 `RecursionError` 无限递归 | `RotaryEmbedding.__init__` 用 rope 初始化函数算 `inv_freq` 再 `clone()` 出 `original_inv_freq`，懒化后两者**自引用成环** | 物化前用模块自身 `_compute_inv_freq` 在 CPU 上重建这些非持久化 buffer（数值与官方实现一致） |
| 3 | 保存 checkpoint 时 `IndexError` | LoRA 只注入注意力/共享专家，路由专家状态为空 → `gather_state_dict` 的 P2P ops 为空列表仍进入 `batch_isend_irecv` | `checkpoint_io/utils.py` 两端对称跳过空 ops |
| 4 | 数据加载报 `TypeError`（tokens 里混入字符串） | transformers 5.x 的 `apply_chat_template(return_dict=True)` 返回 `BatchEncoding`，**不再继承 dict**，`isinstance(x, dict)` 判断失效 → 迭代时取到键名 | 鸭子类型判断 `hasattr(x, "keys") and "input_ids" in x` |
| 5 | `NameError: gt_answer` | coati loader 只在 `"messages" in chat` 分支定义变量，普通对话格式走到别处引用 | 分支外提前初始化为 `None` |

### 3.4 适配验证

- `test_ep_moe.py`：EP 模块（ep=1）与原生实现前向数值等价
  （相对误差 0.6%——两条路径 `index_add` 累加顺序不同的正常舍入，非错误）+
  反向传播通过；
- `test_ep_moe_dist.py`：8 卡分布式前向 + 反向 5ms/iter 通过；
- 补丁存档 `patches/0001`（900 行 diff），对应分支
  `KaimingQ/ColossalAI:dsv4-flash-adaptation`。

## 4. 数据管线攻坚

统一数学推理领域（训练/后训练同域，保证针对性）：

| 困难 | 现象 | 解决 |
|---|---|---|
| 流式下载进程异常退出 | 长迭代结束时报 `terminate called without an active exception`（abort，rc=-6），**但文件已完整写出** | 分批子进程模式（每批 500 条、`--skip` 偏移续拉），**按文件内容判定成败**而非进程退出码 |
| 分批重复拉取相同样本 | 每批从流头部重新拉取 | `--skip` 偏移 + 批次文件落盘后合并 |
| HF 缓存 core dump | NFS 上 mmap 触发崩溃 | `HF_DATASETS_CACHE=/tmp/hf_cache` 指本地盘 |
| 下载失败 | HF 镜像端点与代理同时设置互相冲突 | 二者择一，脚本注释说明 |

产出：SFT 10000 条（AI-MO/NuminaMath-CoT，题 + CoT 解答，过滤 >3000 字符样本）；
DPO 偏好对 4257 对（同题多解，按推导完整度/格式构造 chosen/rejected）。

## 5. 训练攻坚实录（按时间线）

### 5.1 LoRA SFT 冒烟（10 轮迭代，每轮一个独立问题）

| 轮次 | 现象 | 根因/解决 |
|---|---|---|
| 1 | 启动即报 `fabric` 缺失 | `--no-deps` 漏依赖 → 补装固化 |
| 2 | `coati` 缺失 | ColossalChat 独立包 → `-e` 安装 |
| 3 | `RawConversationDataset.__init__` 参数错 | coati API 要求 `system_prompt` → 脚本加参数 |
| 4 | `from_config` 注解求值崩溃 | 懒初始化预热导入（见 3.3-1） |
| 5 | `torchao` 版本冲突 | 0.16.0 固化（见 1-4） |
| 6 | NCCL 600s 超时 | EP 全局索引 bug（见 3.2，最难一轮） |
| 7 | 数据集构造报错 | 发布版无 `chat_template` → 补 DeepSeek 风格模板脚本 |
| 8 | `TypeError` 混入字符串 | `BatchEncoding` 兼容（见 3.3-4） |
| 9 | `NameError` | `gt_answer` 初始化（见 3.3-5） |
| 10 | **通过**：31 步 / 4.27s/it / loss 0.78~1.05（正常量级） / 83.5GB/卡 | — |

### 5.2 正式 SFT 调优

| 尝试 | 结果 | 结论 |
|---|---|---|
| seq1024 × bs4 | 反向 OOM（93.8GB 已用 + 需 1.97GB） | 激活翻倍，放弃 |
| seq512 × bs4 + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | rank7 `unhandled cuda error` → 首层 all-to-all 超时 | 该分配器选项与本环境 NCCL a2a **不兼容**，弃用并记录 |
| seq512 × bs2 × acc2（有效批 8） | **通过**：1250 步 / 4.14s/it / loss 1.07→0.39 / 82.1GB/卡 | 最终配置；产物导出 bin + safetensors 双格式 |

### 5.3 DPO/SimPO 插件选型（实测排除法）

| 插件 | 实测结果 |
|---|---|
| `gemini` | GeminiPlugin **显式不支持 LoRA**（启动即拒）→ 排除 |
| `3d`（HybridParallel） | 无专家并行切分，融合专家 568GB 落单卡 → **OOM 实测** → 排除 |
| `moe`（本项目为 `train_dpo.py` 新增该路线） | EP=8 + ZeRO1，与 SFT 同一验证路线 → **通过**：133 步 / 31 分钟 / loss 稳定在 ~1.0 / 87.6GB/卡 |

算法选 **SimPO**（`--disable_reference_model`）：省去第二份 568GB 参考模型，
是 8×96GB 上能完成偏好对齐的关键。

### 5.4 ORPO nan 攻坚（完整定位过程）

**现象**：~14 步出现 `LogBackward0 returned nan values`，lr 5e-6 与 1e-6 均复现。
初期误判为学习率问题（浪费一轮），后改用**一次性诊断探针**——在损失合成处
打印各中间量有限性：

```
[NAN-DBG] step=0 logits_finite=True logp_c_finite=True logp_r_finite=True
          nll=4.9603 nll_dtype=torch.float32 or_loss=nan lor=nan
```

10 分钟锁定：**nan 只来自 odds-ratio 项**。随后逐层剥离出三个叠加根因：

| # | 根因（代码机制） | 修复 |
|---|---|---|
| 1 | `OddsRatioLoss` 用 `ratio = log(sigmoid(Δ))`：284B 对数学解答置信度高，Δ（chosen 与 reject 的平均 log-odds 差）常 < -100，sigmoid 在 fp32 下溢为 0 → `log(0) = -inf` | `-softplus(-Δ)`（数学恒等 `log σ(x)`），恒有限、梯度 ∈ (0,1) |
| 2 | **最致命**：`calc_masked_log_probs` 把 mask 乘进 logp，被 mask 位置恰为 0 → `log1p(-exp(0)) = log1p(-1) = -inf` → `-inf × 0 = nan`，**每步必触发**。先试 `torch.where` 选分支：前向正常但**反向仍漏 -inf 梯度**（where 的梯度会流进未选中分支的被选值计算图） | `logp.clamp(max=-1e-4)`：仅影响近概率 1 的 token，梯度 `1/(1-exp(logp))` 有界（~1e4，由 grad_clip 兜底） |
| 3 | 修完前两项后第 1 步仍 nan：探针显示 nll 也异常 → trainer 直接取模型内 **bf16 交叉熵**（传 `labels`）作为 chosen NLL，大 logits 溢出 | 不传 `labels`，与 SimPO 路线一致**由 fp32 logits 重算 CE** |

配套修复：
- `log(-exp(logp) + 1.0001)` 的 epsilon hack 在 logp→0 处有 ~1e4 梯度尖峰 →
  换精确 `log1p(-exp(logp))`；
- 移除 `torch.autograd.set_detect_anomaly(True)`（每步反向慢一倍，且是
  nan 警告的噪音源，干扰定位）；
- `--save_interval 200` → 每 50 优化步写一次 530GB EP 分片，NFS 上
  **`File ... cannot be opened`（rank4）崩溃** → 改 600（> 532 micro 步），
  仅末尾保存。

**结果**：133 步全程零 nan / 40 分钟 / 21.9 s/it / loss ~1.1→~0.5 /
84.7GB/卡，626GB 产物（641 分片）成功保存。

### 5.5 SFT loss 异常偏高攻坚（EP combine 错位 + 报告聚合偏大）

**现象**：LoRA SFT 的 loss 报告值远高于基座模型在 HF 原生前向下的真实 NLL
（同分布数学数据实测 ~0.8），终值仍处于随机预测水平。排查发现两个独立 bug 叠加：

1. **EP combine 缺逆置换（正确性 bug）**：`EpDeepseekV4MoE.ep_experts_forward`
   按本地专家分组计算前用 `local_order` 重排接收序列，回传前未做逆置换，
   源 rank 收到错位的专家输出（每个 token 拿到其他专家的结果）——
   对照 `EpDeepseekV3MoE` 的 `new_x[gatherd_idxs] = outs` 还原步骤，
   V4 移植时遗漏；
2. **报告聚合偏大（显示 bug）**：`lora_finetune.py` 的 `all_reduce_mean`
   原地 all_reduce 求和后只返回商，SFT 分支丢弃返回值直接复用原张量，
   报告的是 8 卡之和而非均值。

**定位过程（逐层排除法）**：
1. 基座模型 HF 评测正确率 0.90 → 权重转换与原生前向无误，问题在训练路径；
2. 层级数值对拍（真实权重单层、8 卡 EP）：`EpDeepseekV4MoE` 输出与原生
   `DeepseekV4SparseMoeBlock` 完全不一致 → 锁定 EP 路径；
3. 二分拆解：单专家计算、本地分组计算均精确 → 问题在 all-to-all dispatch/combine；
4. 复刻 dispatch/combine 的独立脚本逐 pair 对拍通过 → 差异在 `local_order` 重排后的回传；
5. 修复逆置换后层级对拍误差降至 bf16 噪声级（0.06）；
6. 训练首步 loss 已正常但仍偏离基线 → 手工重算 CE 与 `model.loss` 一致，
   差异仅在 tqdm 报告聚合 → 定位到 `all_reduce_mean` 丢弃返回值。

**修复**：combine 前 `unsorted[local_order] = local_out` 恢复接收顺序；
`all_reduce_mean` 改为原地除以组大小。修复后冒烟与正式重训 loss 均与
HF 基线一致（见 5.2 与 `docs/06_report.md`）。

## 6. 工程环境攻坚（NFS / GPU 运维）

| 困难 | 现象 | 对策 |
|---|---|---|
| NFS 写缓存延迟 | 编辑后读回"丢失"；容器内偶见旧版脚本；后台日志"消失" | 编辑后**回读验证**；关键脚本用进程内重写兜底；后台任务日志先写容器本地盘再回读 |
| 宿主机无权删容器内文件 | `Permission denied`（root 写入） | 一律经 `docker exec` 操作 |
| `pkill -f` 误杀自身 | 清理脚本自杀（exit 137） | 模式精确匹配目标脚本名 |
| GPU 僵尸进程 | 杀不掉的 torchrun 残留，显存占满 | `docker restart` + 重跑 `setup_env.sh` 的标准恢复流程 |
| 训练与评测并发 | 评测占卡导致训练 OOM | GPU 独占任务严格串行 |
| 空等已停止的任务 | 长时间 sleep 浪费 | 每次检查先 `ps grep -c` 确认进程存活 |

## 7. 成果与框架优势量化

| 阶段 | 数据/步数 | 吞吐 | 单卡峰值显存 | 收敛 |
|---|---|---|---|---|
| LoRA SFT | 10k×2ep / 1250 步 | 4.14 s/it（约 86 分钟） | 83.7GB / 96GB | loss 1.07 → 0.39 |
| DPO/SimPO | 4257 对 / 133 步 | 14.2 s/it（31 分钟） | 87.6GB / 96GB | loss 稳定在 ~1.0 |
| ORPO | 4257 对 / 133 步 | 21.9 s/it（40 分钟） | 84.7GB / 96GB | loss ~1.1 → ~0.5，零 nan |

ColossalAI 优势体现：
1. **省显存**：568GB 权重 + 激活/优化器状态装进 768GB 总显存（朴素方案单卡
   连 1.36 层都放不下）——核心是 EP=8 专家切分 + 逐专家键布局（每卡只加载/
   保存 1/8 专家）+ ZeRO-1 + 无参考模型算法选型；
2. **训练高效**：三阶段合计约 2.5 小时完成全流程，EP all-to-all 与
   ZeRO 优化器叠加，无需人工干预通信细节；
3. **全流程覆盖与新架构快速接入**：SFT → SimPO → ORPO 一条框架内链路；
   全新架构（V4）的接入成本约 800 行补丁（policy + EP 模块 + 脚本适配）。

## 8. 可复用经验清单

1. 量化权重转换必须**键集合双向断言 + 小规模端到端生成验证**双保险，
   数值误差为 0 不代表键映射正确；
2. 官方发布物可能有隐性残缺（缺张量、config 与实际不符），以模型定义与实际
   张量为准；
3. 显存顶满不是"慢"而是**死锁**（ORPO bs2 时 97/96GB 表现为首步后挂起而非
   OOM 报错），批量实验预留 ≥10% 显存余量；
4. nan 定位用**一次性诊断探针**（打印各中间量有限性），优于
   `detect_anomaly`（慢一倍且是警告噪音源）；
5. `-inf×0`、`log(0)`、bf16 大 logits 溢出是偏好优化算法的三大数值陷阱；
   `torch.where` 不能阻断反向中的 -inf 梯度传播，要用 `clamp`；
6. NCCL 集合超时的第一嫌疑往往是**某个 rank 更早崩溃**，先看全部 rank 的
   最早错误而不是通信栈；
7. 集合通信的"卡死"用**最小分布式复现脚本**定位（小模型绕开加载时间）；
8. 分配器选项（如 `expandable_segments`）与集合通信可能冲突，启用前小规模验证；
9. GPU 独占任务严格串行；后台任务检查前先确认进程存活；
10. NFS 上大文件写入控制频率（检查点过密会崩），编辑要回读验证；
11. EP dispatch/combine 是顺序敏感的双向操作：分组计算后的回传必须做
    **逆置换恢复接收顺序**，上线前用真实权重做层级前向对拍（随机权重测不出错位）；
12. 分布式 loss 报告要用返回值或确认原地语义，否则"loss 异常"可能是
    报告口径错误而非训练错误——先手工重算 CE 对比再怀疑前向。

## 9. 细节优化实验：用满显存余量换吞吐（实测）

**目标**：正式配置（seq512 × bs2 × acc2）峰值 82.1GB/卡，距 96GB 上限有约 14GB
余量。在**不开 CPU offload**（offload 实测吞吐代价过大，弃用）的前提下，
用冒烟脚本（`run_smoke.sh` 参数化：`SMOKE_BS/SMOKE_ACC`）逐档加大批量，
寻找吞吐最优点。

**指标口径**：吞吐按每优化步处理的 token 数折算（有效批 = bs × acc × 8 卡，
seq=512）。

| 配置 | 有效批 | 每优化步耗时 | 折算吞吐（tok/s/卡） | 单卡峰值显存 |
|---|---|---|---|---|
| bs2 × acc1（基线/冒烟默认） | 16 | 4.14 s（正式训练稳态） | ~14.9 | 82.1GB |
| bs4 × acc2 | 64 | 10.62 s | ~15.1 | 84.8GB |
| bs6 × acc2 | 96 | 18.17 s | **~16.0** | 87.9GB |
| bs8 × acc2 | 128 | OOM | — | 89.1GB 已用 + 需 1.97GB |

（实验条件：64 条样本，其余超参与正式训练一致；显存为 `Max device memory
usage` 日志值。）

**结论**：
1. 批量从 2 提到 6，吞吐提升约 **7%**，显存仅增 5.8GB（激活相对权重占比小，
   这正是 EP 切分后权重主导显存结构的体现）；
2. **bs8 触顶**：反向需再分配 1.97GB 而单卡仅剩 ~1.3GB，OOM——bs6 与 bs8 之间
   即为 96GB 的实际上限（另注：错误提示的 `expandable_segments` 方案在本环境
   与 NCCL a2a 不兼容，见 5.2，不可用）；
3. 正式训练追求稳健取 bs2×acc2（82.1GB，余量 14GB）；吞吐优先场景推荐
   **bs6×acc2**（87.9GB，仍留有 ≥8GB 安全垫）：`SMOKE_BS=6 SMOKE_ACC=2` 可复现。

## 10. EP MoE 训练吞吐优化（torch.profiler 定位，参照 Megatron-Core / DeepEP 思路）

**方法**：给 `lora_finetune.py` 接入 `torch.profiler`（`PROFILE_STEPS` 环境变量控制，
wait=1/warmup=2/active=N 窗口，rank0 导出 chrome trace，采完提前退出）。
4 步采样显示 GPU busy 仅 65%，三大瓶颈：

| 瓶颈 | 证据（4 步窗口） | 占比 |
|---|---|---|
| GPU→CPU 同步风暴 | `_local_scalar_dense` 4.4 万次、`cudaMemcpyAsync` 5.8 万次、`cudaStreamSynchronize` 2.5s；EP 专家循环内 `int(local_counts[e])` + `DPGradScaler` 的 tensor assert 每层 ~100 次同步 | GPU 空闲 35% 主因 |
| NCCL all_to_all 等待 | 1720 次 ×2.7~4ms，rank 间方差 30%~45%（同步导致各 rank 到达集合点相位不齐） | GPU 时间 30~45% |
| 专家小 GEMM + eager attention 物化 | `aten::mm` 5.2 万次 ×47~68µs；MQA `repeat_kv` 64 倍拷贝 + `[B,64,S,S]` 物化 | 16~24% / ~20% |

**优化**（`colossalai/shardformer/modeling/deepseek_v4.py`）：
1. **同步消除**：`local_counts.tolist()`/`activate_experts.tolist()` 每层各 1 次批量取回，
   替代循环内逐专家同步；
2. **grouped GEMM**（Megatron `TEGroupedMLP` 思路）：`fuse_local_experts` 在权重加载后将本地
   32 专家融合为 3D 冻结权重，forward 用 `torch._grouped_mm` 一次计算（每组 pad 到 8 的倍数
   满足 cutlass 16B 对齐，平均 ~11% 冗余计算）；fuse 前两阶段清理 optimizer/ZeRO 容器对原
   参数的引用链（HybridAdam 全量收集 + `pg_to_param_list` + `param_to_pg`）并打断
   LazyTensor `tolist` 自引用环——否则原参数无法释放导致 OOM；
   `unfuse_local_experts` 供完整 EP 分片保存前还原布局；
3. **attention 等价提速**（head_dim=512 无法走 SDPA/FA）：`v4_fast_eager_attention` 用 MQA
   matmul 广播免 `repeat_kv` 64 倍拷贝，sinks 并入 softmax 的 max/分母免 `[.., S+1]` cat；
   单测对拍与 HF eager 差异为 bf16 噪声级，fwd 5.21→3.91ms。

**效果**（SFT 冒烟 4 步 profile 窗口，loss 完全一致 0.892~0.893）：

| 指标 | 基线 | 同步消除 | +grouped GEMM/attention |
|---|---|---|---|
| 每步耗时 | 5.60s | 4.30s | **3.38s** |
| `nccl all_to_all` | 4.59s | 3.22s | **1.34s（-71%）** |
| kernel launch 数 | 51.3 万 | 48.1 万 | **37.7 万** |

实际端到端：LoRA SFT 4.14→**3.11 s/it**（-25%），DPO/SimPO 全量重训 14.2→**11.26 s/it**
（133 步 25 分 13 秒，loss 0.997 与基线一致，fuse/unfuse 保存路径验证通过）。
单测 `scripts/test_grouped_forward.py`（空组/任意 counts/8 倍数边界）ALL PASS。
