#!/usr/bin/env bash
# 容器内环境安装(幂等): 从挂载源码安装已适配的 ColossalAI
# 用法: docker exec -it dsv4-colossal-train bash -c "bash scripts/setup_env.sh"
set -euo pipefail

# 若所在环境访问 PyPI/GitHub 需要代理, 请在此处按实际情况设置, 例如:
#   export https_proxy=http://<proxy-host>:<port>
#   export http_proxy=http://<proxy-host>:<port>

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COLOSSAL_SRC="${PROJECT_ROOT}/third_party/ColossalAI"

echo "[setup] 1/3 环境信息"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

echo "[setup] 2/3 安装 ColossalAI (源码, --no-deps)"
# 官方 requirements 钉住 torch<=2.5.1 / transformers==4.51.3,
# 与最新 PyTorch + DeepSeek-V4 (transformers>=5.0) 冲突,
# 故用 --no-deps 安装, 依赖由镜像 Dockerfile 与本脚本显式管理,
# 适配补丁记录在 patches/ 与 docs/02_adaptation.md
pip install -e "${COLOSSAL_SRC}" --no-deps --no-build-isolation

echo "[setup] 2b/3 安装 ColossalChat (coati: SFT/DPO/ORPO 训练脚本依赖)"
pip install -e "${COLOSSAL_SRC}/applications/ColossalChat" --no-deps

echo "[setup] 3/3 补齐 colossalai 运行时依赖(版本不受官方钉死限制)"
pip install numpy psutil packaging rich click contexttimer \
    einops pydantic safetensors sentencepiece protobuf \
    "ray" "peft" "galore_torch" fastapi uvicorn \
    fabric google rpyc || true

# peft 要求 torchao>=0.16 (NGC 基础镜像自带 0.11 不兼容)
pip install --no-deps "torchao==0.16.0" || true

python -c "import colossalai; print('colossalai', colossalai.__version__)"
python -c "import transformers; print('transformers', transformers.__version__)"
echo "[setup] 完成"
