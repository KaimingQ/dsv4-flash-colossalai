# 从零复现指南（完整命令序列）

> 新机器上按本文顺序执行即可复现全流程。前置：8×H20 单机、Docker、存放模型权重的目录
> （全流程以环境变量 `MODEL_ROOT` 指定，内含官方发布的 `deepseek-v4-flash-0731`）、
> 可访问 GitHub/PyPI/HuggingFace 的网络（如需代理，请先设置 `http(s)_proxy`）。

## 0. 设置模型目录

```bash
export MODEL_ROOT=/path/to/model-dir   # 模型权重所在目录, 全流程脚本统一读取
```

## 1. 拉取代码与第三方源码

```bash
git clone https://github.com/KaimingQ/dsv4-flash-colossalai.git
cd dsv4-flash-colossalai
mkdir -p third_party

# 官方最新 ColossalAI (基线 commit 4f9953b, main)
git clone --depth 1 https://github.com/hpcaitech/ColossalAI.git third_party/ColossalAI

# 应用 DeepSeek-V4 适配补丁
cd third_party/ColossalAI
git apply ../../patches/0001-deepseek_v4-ep-policy-and-lora-adaptation.patch
cd ../..

# (可选) FlagOS 官方权重转换工具, 用于 0731 原版 (FP4 专家) 转换
git clone --depth 1 https://github.com/flagos-ai/DeepSeek-V4-FlagOS.git third_party/DeepSeek-V4-FlagOS
```

## 2. 构建镜像并启动容器

```bash
bash docker/build.sh          # 基础镜像缺失时直接拉取; 可设 MIRROR_REGISTRY 走镜像加速
bash docker/run.sh            # 启动 8 卡容器
docker exec -it dsv4-colossal-train bash -c "bash scripts/setup_env.sh"
```

验证：`torch 2.8.0a0 / transformers 5.15.1 / colossalai 0.5.0 / 8 GPU`。

## 3. 权重转换（FP4/FP8 → BF16，一次性）

源：`$MODEL_ROOT/deepseek-v4-flash-0731`（官方发布，专家 FP4 + 非专家 FP8）
产物：
- `DeepSeek-V4-Flash-BF16-v2`（逐专家键布局，训练用，约 530GB）
- `DeepSeek-V4-Flash-BF16-v2-fused`（融合 3D 布局，HF 推理加载用）

```bash
# 反量化算法来自 FlagOS 官方工具 (third_party/DeepSeek-V4-FlagOS/convert_weight.py)
# v2 转换: 完整键映射 + 与 HF 模型定义严格双向校验 (后台, 约数十分钟)
docker exec -d dsv4-colossal-train bash -c \
    "cd /workspace && python scripts/convert_native_to_bf16_v2.py \
        --output $MODEL_ROOT/DeepSeek-V4-Flash-BF16-v2 > logs/convert_v2.log 2>&1"

# 补 chat_template (原生发布版未提供)
docker exec dsv4-colossal-train python scripts/add_chat_template.py \
    "$MODEL_ROOT/DeepSeek-V4-Flash-BF16-v2"

# 融合为 3D 布局 (下游推理以 HF from_pretrained 直接加载)
docker exec dsv4-colossal-train bash -c \
    "cd /workspace && python scripts/merge_ep_shards_to_hf.py \
        --input $MODEL_ROOT/DeepSeek-V4-Flash-BF16-v2 \
        --output $MODEL_ROOT/DeepSeek-V4-Flash-BF16-v2-fused"
```

重要背景（详见 `docs/02_adaptation.md` 踩坑记录）：
- 不要用 transformers 自动映射转换本模型（`hc_*_scale` 等条目缺失 → 随机初始化 → 生成乱码）；
- 纯 FP8 变体（DeepSeek-V4-Flash-FP8）的 checkpoint 缺失全部 43 层 `wo_a.scale`，无法完整反量化。

## 4. 适配单元测试

```bash
docker exec dsv4-colossal-train python scripts/test_ep_moe.py
# 预期: autopolicy 解析 OK / EP 模块与原生数值一致 / 反向传播 OK
```

## 5. 数据准备（统一数学领域）

```bash
docker exec dsv4-colossal-train bash -c "MAX_SAMPLES=10000 bash scripts/prepare_data.sh"
```

## 6. LoRA SFT

```bash
docker exec dsv4-colossal-train bash scripts/run_smoke.sh        # 冒烟: 64 条验证全链路
docker exec dsv4-colossal-train bash scripts/run_full_sft.sh     # 正式: 全量数据
```

## 7. 后训练（DPO/SimPO 与 ORPO）

```bash
docker exec dsv4-colossal-train bash -c "cd /workspace && bash scripts/run_dpo.sh"   # DPO/SimPO (无参考模型)
docker exec dsv4-colossal-train bash -c "cd /workspace && bash scripts/run_orpo.sh"  # ORPO (无参考模型)
```

## 8. 产物验证与导出

```bash
# LoRA adapter 结构/参数验证
docker exec dsv4-colossal-train python scripts/verify_adapter.py --adapter output/dsv4-lora-sft-full/lora
# EP 分片 -> HF 融合布局 (下游推理加载)
docker exec dsv4-colossal-train bash -c "cd /workspace && python scripts/merge_ep_shards_to_hf.py \
    --input output/dsv4-dpo/modeling --output output/dsv4-dpo-hf"
```

详见 `docs/`（01 环境 / 02 适配 / 03 数据 / 04 SFT / 05 后训练 / 06 报告）。

## 常见问题速查

| 现象 | 原因/处理 |
|---|---|
| `datasets` core dump | HF 缓存在 NFS 上，设 `HF_DATASETS_CACHE=/tmp/hf_cache` |
| 容器内新文件"不存在" | NFS 写缓存延迟，稍等或 `touch` 后重试 |
| fp8 推理内核报错 | `kernels` 与 triton 3.3 不兼容；训练走 BF16 不受影响 |
| 转换产物全错（值域±448） | 加载时误删 `quantization_config`，见 02 文档踩坑 |
