#!/usr/bin/env bash
# 正式训练: 284B LoRA SFT 全量数学数据 (容器内执行)
# 显存预算 (实测, 见 docs/07_improvements.md 第 9 节):
#   默认 bs2×acc2: 82.1GB/卡 (余量 14GB, 稳健)
#   吞吐优先 bs6×acc2: 87.9GB/卡, 吞吐 +7% (SFT_BS=6 SFT_ACC=2 启用)
#   bs8 已实测 OOM, 勿超过 bs6
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

# 注: expandable_segments 实测与本环境 NCCL a2a 不兼容 (unhandled cuda error), 不启用

MODEL=${MODEL:-"${MODEL_ROOT:?需先设置 MODEL_ROOT(模型权重所在目录)}/DeepSeek-V4-Flash-BF16-v2"}
DATASET=${DATASET:-data/sft_public.jsonl}
SAVE_DIR=${SAVE_DIR:-output/dsv4-lora-sft-full}
TB_DIR=${TB_DIR:-logs/tb-sft-full}
SFT_BS=${SFT_BS:-2}
SFT_ACC=${SFT_ACC:-2}

TRAIN_SCRIPT=third_party/ColossalAI/applications/ColossalChat/examples/training_scripts/lora_finetune.py

colossalai run --nproc_per_node 8 "${TRAIN_SCRIPT}" \
    --pretrained "${MODEL}" \
    --dataset "${DATASET}" \
    --plugin moe \
    --ep 8 \
    --zero_stage 1 \
    ${OFFLOAD:+--zero_cpu_offload} \
    --low_cpu_mem \
    --mixed_precision bf16 \
    --lr 2e-5 \
    --max_length 512 \
    --batch_size "${SFT_BS}" \
    --accumulation_steps "${SFT_ACC}" \
    --lora_rank 16 \
    --lora_alpha 32 \
    --num_epochs 2 \
    --warmup_steps 20 \
    --grad_clip 1.0 \
    --use_grad_checkpoint \
    --tensorboard_dir "${TB_DIR}" \
    --save_dir "${SAVE_DIR}" 2>&1 | tee logs/full_run.log
