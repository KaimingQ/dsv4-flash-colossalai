#!/usr/bin/env bash
# 构建训练镜像。若基础镜像缺失, 直接拉取; 如环境配有镜像仓库加速器,
# 可设置 MIRROR_REGISTRY 环境变量 (如 image-cloud.example.com) 自动加前缀拉取。
set -euo pipefail
cd "$(dirname "$0")"

BASE_IMAGE=nvcr.io/nvidia/pytorch:25.06-py3
IMAGE_NAME=dsv4-flash-colossalai:latest

if ! docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
    if [[ -n "${MIRROR_REGISTRY:-}" ]]; then
        echo "[build] 基础镜像缺失, 从镜像仓库 ${MIRROR_REGISTRY} 拉取..."
        docker pull "${MIRROR_REGISTRY}/${BASE_IMAGE}"
        docker tag "${MIRROR_REGISTRY}/${BASE_IMAGE}" "${BASE_IMAGE}"
    else
        echo "[build] 基础镜像缺失, 直接拉取..."
        docker pull "${BASE_IMAGE}"
    fi
fi

docker build --network=host -t "${IMAGE_NAME}" .
echo "[build] 完成: ${IMAGE_NAME}"
