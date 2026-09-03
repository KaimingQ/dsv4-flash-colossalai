#!/usr/bin/env bash
# GPU 空闲后自动依次重跑 ORPO 与 LoRA SFT 全量 (容器内执行, nohup 后台):
# 吞吐优化后的端到端重测; 每卡显存占用 <5GB 视为空闲, 每 2 分钟轮询
set -uo pipefail
cd /workspace

wait_gpu_free() {
  while true; do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    if [[ "${USED:-99999}" -lt 5000 ]]; then
      echo "[$(date '+%F %T')] GPU free (max used ${USED} MiB), starting next run"
      return 0
    fi
    sleep 120
  done
}

wait_gpu_free
echo "[$(date '+%F %T')] === ORPO rerun start ==="
MODEL=/home/shared/deepseek-ai/DeepSeek-V4-Flash-BF16-v2 bash scripts/run_orpo.sh || {
  echo "[$(date '+%F %T')] ORPO FAILED"; exit 1; }
echo "[$(date '+%F %T')] === ORPO done, SFT rerun start ==="

wait_gpu_free
MODEL=/home/shared/deepseek-ai/DeepSeek-V4-Flash-BF16-v2 bash scripts/run_full_sft.sh || {
  echo "[$(date '+%F %T')] SFT FAILED"; exit 1; }
echo "[$(date '+%F %T')] === all reruns done ==="
