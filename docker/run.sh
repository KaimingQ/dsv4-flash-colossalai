#!/usr/bin/env bash
# 启动训练容器: 8 卡全量挂载 + 模型权重目录 + 项目目录
# 需先设置 MODEL_ROOT (模型权重所在目录), 例如:
#   export MODEL_ROOT=/path/to/model-dir
set -euo pipefail
cd "$(dirname "$0")"

PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
: "${MODEL_ROOT:?需先设置 MODEL_ROOT (模型权重所在目录), 例如: export MODEL_ROOT=/path/to/model-dir}"
IMAGE_NAME=dsv4-flash-colossalai:latest
CONTAINER_NAME=dsv4-colossal-train

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

# 模型目录按原路径挂载, 容器内外路径一致, 训练脚本无需改动
docker run -d --name "${CONTAINER_NAME}" \
    --gpus all \
    --network=host \
    --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --shm-size=64g \
    -e MODEL_ROOT="${MODEL_ROOT}" \
    -v "${MODEL_ROOT}":"${MODEL_ROOT}" \
    -v "${PROJECT_DIR}":/workspace \
    -w /workspace \
    "${IMAGE_NAME}" \
    bash -c "sleep infinity"

echo "[run] 容器已启动: ${CONTAINER_NAME} (MODEL_ROOT=${MODEL_ROOT})"
echo "[run] 进入容器: docker exec -it ${CONTAINER_NAME} bash"
echo "[run] 容器内先执行: bash scripts/setup_env.sh"
