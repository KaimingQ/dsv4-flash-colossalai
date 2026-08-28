#!/usr/bin/env bash
# 数据集下载与格式转换 (容器内执行)
# 统一领域: 数学推理 (训练与后训练同一领域, 保证训练效果有针对性)
#   SFT : AI-MO/NuminaMath-CoT           -> data/sft_public.jsonl
#   DPO : NuminaMath 多解质量偏好对      -> data/dpo_public (arrow)
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

# 若所在环境访问 HuggingFace 需要代理, 请按实际情况在此设置:
#   export https_proxy=http://<proxy-host>:<port>
#   export http_proxy=http://<proxy-host>:<port>

MAX_SAMPLES=${MAX_SAMPLES:-5000}

echo "[data] 1/2 SFT 数学数据 (NuminaMath-CoT, ${MAX_SAMPLES} 条)"
python scripts/convert_math_sft_data.py --max_samples "${MAX_SAMPLES}" --output data/sft_public.jsonl

echo "[data] 2/2 DPO 数学偏好数据"
python scripts/convert_math_dpo_data.py --max_samples "${MAX_SAMPLES}" --output data/dpo_public

echo "[data] 完成: $(ls -lh data/ | tail -n +2)"
