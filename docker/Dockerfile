# =====================================================================
# DeepSeek-V4-Flash (284B) x ColossalAI 8xH20 训练镜像
# 基础镜像: NVIDIA NGC PyTorch
#   直接拉取: docker pull nvcr.io/nvidia/pytorch:25.06-py3
#   (若所在环境配有镜像仓库加速器, 可通过仓库前缀拉取同名镜像)
# 说明: 内含最新稳定 PyTorch + CUDA, NGC 镜像已预装 flash-attn
# =====================================================================
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.06-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# ColossalAI 官方 requirements 声明 torch>=2.2.0,<=2.5.1,
# 但本实验目标适配最新 PyTorch (NGC 25.06 内含 2.8+/2.9),
# 因此安装时放开版本约束, 兼容性验证结果记录在 docs/02_adaptation.md

# ---- Python 依赖 (不含 colossalai 本体; 本体由启动脚本从挂载源码安装, 便于适配迭代) ----
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        "transformers>=5.0.0" \
        peft \
        accelerate \
        "datasets" \
        loralib \
        galore_torch \
        tensorboard \
        tqdm jsonlines sentencepiece tiktoken tokenizers \
        gpustat packaging ninja \
        "torchao==0.16.0" \
        wandb \
    # 注: pyext 不兼容 Python 3.12 (inspect.getargspec 已移除), 故不安装
    && pip check || true

# ---- 工作目录 ----
WORKDIR /workspace

# 容器启动默认进入 bash
CMD ["/bin/bash"]
