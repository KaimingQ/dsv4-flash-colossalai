#!/usr/bin/env bash
# LoRA SFT 冒烟/调优测试 (容器内执行): 默认 64 条样本 1 epoch, 验证全链路:
# 模型加载 -> EP 切分 -> LoRA 注入 -> 前向/反向 -> loss 下降 -> 无 OOM
# 可用环境变量调参做吞吐/显存实验: SMOKE_BS / SMOKE_ACC / SMOKE_SAMPLES
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

MODEL=${MODEL:-"${MODEL_ROOT:?需先设置 MODEL_ROOT(模型权重所在目录)}/DeepSeek-V4-Flash-BF16-v2"}
SMOKE_BS=${SMOKE_BS:-2}
SMOKE_ACC=${SMOKE_ACC:-1}
SMOKE_SAMPLES=${SMOKE_SAMPLES:-64}
SMOKE_DATA=data/sft_smoke.jsonl
[[ -f "${SMOKE_DATA}" ]] || head -"${SMOKE_SAMPLES}" data/sft_public.jsonl > "${SMOKE_DATA}"

TRAIN_SCRIPT=third_party/ColossalAI/applications/ColossalChat/examples/training_scripts/lora_finetune.py

colossalai run --nproc_per_node 8 "${TRAIN_SCRIPT}" \
    --pretrained "${MODEL}" \
    --dataset "${SMOKE_DATA}" \
    --plugin moe \
    --ep 8 \
    --zero_stage 1 \
    ${OFFLOAD:+--zero_cpu_offload} \
    --low_cpu_mem \
    --mixed_precision bf16 \
    --lr 2e-5 \
    --max_length 512 \
    --batch_size "${SMOKE_BS}" \
    --accumulation_steps "${SMOKE_ACC}" \
    --lora_rank 16 \
    --lora_alpha 32 \
    --num_epochs 1 \
    --warmup_steps 2 \
    --grad_clip 1.0 \
    --use_grad_checkpoint \
    --tensorboard_dir logs/tb-smoke \
    --save_dir output/smoke-lora 2>&1 | tee logs/smoke_run.log
