# 04 LoRA SFT（284B @ 8×H20）

> 284,332,230,231 参数的 DeepSeek-V4-Flash 在单机 8×H20（768GB 显存 + 2TB 内存）上的
> LoRA 监督微调。权重来源：0731 官方发布 → v2 转换管线（见 02 文档）。

## 1. 训练配置

| 项 | 值 |
|---|---|
| 插件 | `--plugin moe`（MoeHybridParallelPlugin），EP=8，ZeRO stage 1 |
| 模型布局 | 逐专家键（23215 键），每 rank 仅物化/加载 1/8 路由专家 |
| 精度 | bf16 混合精度 + 梯度检查点 |
| LoRA | r=16, alpha=32，target：q_a_proj/q_b_proj/kv_proj/o_b_proj + 共享专家 gate/up/down |
| 数据 | NuminaMath-CoT 数学 10000 条 × 2 epochs（统一数学领域） |
| 优化器 | HybridAdam，lr 2e-5，cosine + warmup，seq512，bs2×acc2（有效批 8） |

## 2. 显存账（凸显"省显存"）

284B bf16 权重 = 568GB，逼近 8 卡显存总和（768GB）——朴素数据并行不可行。本方案：

| 组成 | 每卡占用 |
|---|---|
| 注意力/共享专家/embed（bf16，全量） | ~14GB |
| 路由专家（554GB / 8 卡 EP 切分） | ~69GB |
| LoRA 参数 + 优化器状态（可训练参数仅 0.02%） | <1GB |
| 激活（梯度检查点） | 数 GB |

对比基线（同 284B，8×96GB）：
- HF transformers + PEFT：单卡需 568GB 权重 → **不可行**；
- DeepSpeed ZeRO-3：568GB/8=71GB/卡 权重可行，但专家无 EP 感知（全复制后切分，
  通信量大、无专家级负载感知）；
- ColossalAI moe plugin：专家并行原生切分 + ZeRO 组合，实测 ~84GB/卡 **端到端跑通**。

## 3. 适配踩坑与修复（全部沉淀为补丁 `patches/0001-*`）

1. 融合 3D 专家张量无法部分加载 → 逐专家键布局 + ModuleList 改造；
2. 置空非持有专家后计算索引错位 → 保持全局索引布局（此为 all-to-all 超时假象的根因）；
3. transformers 5.x `torch.LongTensor | None` 注解在懒初始化下求值崩溃 → 预热导入；
4. RotaryEmbedding 懒链自引用环 → `_fix_lazy_rotary_buffers` 重建 buffer；
5. coati `apply_chat_template` 5.x BatchEncoding 兼容；
6. `o_a_proj`（GroupedLinear 5D 输入）排除在 LoRA target 之外；
7. 保存阶段空专家状态 P2P ops 崩溃 → 空列表保护。

## 4. 正式训练结果（10000 条 × 2 epochs = 1250 步）

配置：`scripts/run_full_sft.sh`。

| 指标 | 值 |
|---|---|
| loss | 起始 → 终值（见 `logs/full_run.log` / 报告图） |
| 吞吐 | ~4.15 s/it（稳态） |
| 显存峰值 | ~84GB/卡（96GB 的 88%） |
| 产物 | `output/dsv4-lora-sft-full/lora/`（PEFT adapter，bin+safetensors 双格式） |

调优记录（如实）：
- seq1024×bs4：反向 OOM（93.8GB 已用 + 需 1.97GB）→ 放弃；
- seq512×bs4 + `expandable_segments`：rank7 unhandled cuda error（该分配器选项与本环境
  NCCL all-to-all 不兼容）→ 放弃；
- seq512×bs2×acc2：稳定跑完。
