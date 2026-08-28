#!/usr/bin/env bash
# M6 DPO/SimPO 后训练 (容器内执行)
# 284B 显存策略: moe 插件 (EP=8 专家切分 + ZeRO1) + LoRA + SimPO
# 注: GeminiPlugin 不支持 LoRA; 3d 插件无专家切分 (融合专家 568GB 落单卡会 OOM);
#     moe 插件复用 SFT 验证过的 EP 路线; SimPO 无参考模型省去第二份 568GB
# 数据: 数学领域偏好对 (NuminaMath 多解质量排序构造)
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

# 逐专家键布局 (EP 按需加载专家)
MODEL=${MODEL:-"${MODEL_ROOT:?需先设置 MODEL_ROOT(模型权重所在目录)}/DeepSeek-V4-Flash-BF16-v2"}
SAVE_DIR=${SAVE_DIR:-output/dsv4-dpo}

TRAIN_SCRIPT=third_party/ColossalAI/applications/ColossalChat/examples/training_scripts/train_dpo.py

colossalai run --nproc_per_node 8 "${TRAIN_SCRIPT}" \
    --pretrain "${MODEL}" \
    --dataset data/dpo_public \
    --plugin moe \
    --ep 8 \
    --zero_stage 1 \
    --disable_reference_model \
    --loss_type simpo_loss \
    --gamma 0.5 \
    --beta 2.0 \
    --lora_config configs/lora_dpo.json \
    --mixed_precision bf16 \
    --lr 5e-6 \
    --max_length 512 \
    --batch_size 2 \
    --accumulation_steps 2 \
    --max_epochs 1 \
    --grad_clip 1.0 \
    --grad_checkpoint \
    --save_interval 200 \
    --save_dir "${SAVE_DIR}" \
    --config_file output/dpo_config.json 2>&1 | tee logs/dpo_run.log
