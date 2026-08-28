# DeepSeek-V4-Flash (284B) × ColossalAI × 8×H20：显存受限下的大模型后训练全流程

在单机 8×NVIDIA H20（96GB/卡）上，从零适配官方最新 ColossalAI 到 DeepSeek-V4 新架构，
完整跑通 284B MoE 模型的 **权重转换 → LoRA SFT → DPO/SimPO → ORPO** 后训练链路：
三个阶段全部收敛、单卡峰值显存 ≤85GB。展示大模型训练框架的核心价值：
**build high-quality private models at low cost**——省显存、训练高效、
覆盖训练/微调/后训练全流程。

## 为什么需要分布式框架（显存账本）

| 项 | 大小 | 说明 |
|---|---|---|
| 模型参数（bf16） | 568GB | 284.33B × 2B，**占 8 卡显存总和 768GB 的 74%** |
| 单卡可承载（朴素复制） | 1.36 层 | 不分片完全不可行 |
| 本项目方案 | EP=8 + ZeRO-1 + 逐专家键布局 | 每卡仅驻留 1/8 路由专家，实测峰值 **≤85GB/卡** |

## 实验主线与结果（全部实测）

训练与后训练统一**数学推理**领域数据（效果有针对性）：

| 阶段 | 数据 | 收敛情况 | 单卡峰值显存 |
|---|---|---|---|
| LoRA SFT（1250 步，4.14s/it） | NuminaMath-CoT 10k×2ep | loss 45.7 → 10.4 | 82.1GB / 96GB |
| DPO/SimPO（133 步，13.0s/it） | NuminaMath 偏好对 4257 | loss 收敛至 1.2 | 85.5GB / 96GB |
| ORPO（133 步，17.2s/it） | 同上 | loss 5.50 → 4.07，全程零 nan | 84.8GB / 96GB |

> 完整数据、loss/显存曲线图与复现命令见 [`docs/06_report.md`](docs/06_report.md)。

## 目录结构

```
docker/      Dockerfile / build.sh / run.sh（NGC 基础镜像 + 训练依赖）
scripts/     环境安装、权重转换、数据准备、训练启动、产物导出、出图
configs/     LoRA 配置
patches/     ColossalAI DeepSeek-V4 适配补丁（可 git apply 到官方源码）
docs/        01 环境 / 02 适配 / 03 数据 / 04 SFT / 05 后训练 / 06 报告 / 07 改进攻坚记录
data/  logs/  output/   （.gitignore，训练数据/日志/产物）
```

## 快速开始

```bash
# 0. 设置模型权重所在目录 (全流程统一使用)
export MODEL_ROOT=/path/to/model-dir   # 内含官方发布的 deepseek-v4-flash-0731

# 1. 环境（详见 docs/01_environment.md）
bash docker/build.sh && bash docker/run.sh
docker exec -it dsv4-colossal-train bash -c "bash scripts/setup_env.sh"

# 2. 权重（0731 官方 FP4/FP8 -> BF16 逐专家键, 详见 REPRODUCE.md）
docker exec dsv4-colossal-train python scripts/convert_native_to_bf16_v2.py \
    --output "$MODEL_ROOT/DeepSeek-V4-Flash-BF16-v2"
docker exec dsv4-colossal-train python scripts/add_chat_template.py \
    "$MODEL_ROOT/DeepSeek-V4-Flash-BF16-v2"

# 3. 数据（统一数学领域）
docker exec dsv4-colossal-train bash -c "MAX_SAMPLES=10000 bash scripts/prepare_data.sh"

# 4. 训练与后训练
docker exec dsv4-colossal-train bash scripts/run_smoke.sh       # 冒烟: 64 条验证全链路
docker exec dsv4-colossal-train bash scripts/run_full_sft.sh    # LoRA SFT
docker exec dsv4-colossal-train bash scripts/run_dpo.sh         # DPO/SimPO
docker exec dsv4-colossal-train bash scripts/run_orpo.sh        # ORPO

# 5. 产物验证与报告图表
docker exec dsv4-colossal-train python scripts/verify_adapter.py --adapter output/dsv4-lora-sft-full/lora
docker exec dsv4-colossal-train python scripts/make_report_figures.py
```

完整从零复现命令序列：[`REPRODUCE.md`](REPRODUCE.md)。

## 关键技术点

1. **DeepSeek-V4 shardformer 适配**（补丁 `patches/0001`，对应分支
   `KaimingQ/ColossalAI:dsv4-flash-adaptation`）：
   官方 ColossalAI 仅有 DeepSeek-V3 policy；本项目新增 `deepseek_v4` EP policy、
   `EpDeepseekV4MoE`（all-to-all 专家并行）、逐专家键布局改造，
   使每卡只加载/训练 1/8 路由专家——这是 284B 装进 8×96GB 的核心。
2. **FP4/FP8→BF16 权重转换**（`scripts/convert_native_to_bf16_v2.py`）：
   官方发布为推理压缩格式（专家 FP4 + 非专家 FP8），训练需要高精度主权重；
   反量化算法采用 FlagOS 官方实现，完整键映射并与 HF 模型定义严格双向校验
   （23215 键，零缺失零多余）。
3. **ORPO 数值稳定化攻坚**：初版 ~14 步出现 nan，定位为上游实现的三处数值漏洞
   （masked 位置 `log1p(-1)=-inf`、`log(sigmoid)` 下溢、bf16 交叉熵溢出），
   逐一修复后稳定收敛，全记录见 `docs/05_posttraining.md`。
4. **最新栈适配**：PyTorch 2.8 / transformers 5.15（原生 `deepseek_v4`）/ ColossalAI 0.5.0，
   解决懒初始化注解求值、chat_template、5.x tokenizer API 等一系列兼容问题。

## 文档索引

| 文档 | 内容 |
|---|---|
| [docs/01_environment.md](docs/01_environment.md) | 硬件/镜像/容器环境从零搭建 |
| [docs/02_adaptation.md](docs/02_adaptation.md) | V4 架构适配 + 权重转换踩坑全记录 |
| [docs/03_data.md](docs/03_data.md) | 统一数学领域数据管线 |
| [docs/04_lora_sft.md](docs/04_lora_sft.md) | LoRA SFT 配置、显存账、调优实录 |
| [docs/05_posttraining.md](docs/05_posttraining.md) | SimPO/ORPO 后训练 + ORPO nan 攻坚 |
| [docs/06_report.md](docs/06_report.md) | 正式报告：收敛/显存/效率与图表 |
| [docs/07_improvements.md](docs/07_improvements.md) | 改进工作报告：全部困难、定位过程与解决方案细节 |
| [REPRODUCE.md](REPRODUCE.md) | 新机器从零复现的完整命令序列 |
