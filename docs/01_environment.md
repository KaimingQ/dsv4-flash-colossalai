# 01 环境与镜像（从零复现）

> 目标：在 8×NVIDIA H20（96GB/卡）单机上，基于 ColossalAI 最新版完成 DeepSeek-V4-Flash（284B）
> LoRA 微调与后训练全流程。本文档记录从零构建环境的全部步骤。

## 硬件与宿主环境

| 项目 | 配置 |
|---|---|
| GPU | 8 × NVIDIA H20-SXM4（97871 MiB/卡，合计约 768GB） |
| Host 内存 | 2.0 TiB（CPU offload 的关键依托） |
| 存储 | 项目目录与模型目录（`$MODEL_ROOT`）建议放大容量共享存储（本实验为 NFS 14T，注意 mmap/写缓存问题，见踩坑） |
| 系统 | Ubuntu 22.04，Docker 29.2.1，驱动 595.71.05 |

## 网络说明

构建与数据准备需访问 GitHub / PyPI / HuggingFace / NGC。若所在环境需要代理或镜像加速，
请按实际情况在 `scripts/setup_env.sh`、`scripts/prepare_data.sh` 顶部设置
`http_proxy` / `https_proxy`；Docker 基础镜像拉取可通过 `docker/build.sh` 的
`MIRROR_REGISTRY` 环境变量指定镜像仓库前缀。

## 基础镜像

`nvcr.io/nvidia/pytorch:25.06-py3`（NGC 官方 PyTorch 镜像）：

```bash
docker pull nvcr.io/nvidia/pytorch:25.06-py3
```

镜像内含：Python 3.12、PyTorch 2.8.0（NGC 构建）、CUDA 12.9、flash-attn、triton 3.3。

## 构建与启动

```bash
cd docker
export MODEL_ROOT=/path/to/model-dir   # 模型权重所在目录 (全流程统一使用)
bash build.sh   # 构建 dsv4-flash-colossalai:latest（安装训练依赖）
bash run.sh     # 启动容器 dsv4-colossal-train（8 卡、64g shm、挂载 $MODEL_ROOT 与项目目录）
docker exec -it dsv4-colossal-train bash
bash scripts/setup_env.sh   # 容器内：从挂载源码安装已适配的 ColossalAI
```

## 关键版本（实测）

| 组件 | 版本 | 说明 |
|---|---|---|
| PyTorch | 2.8.0a0 (nv25.06) | 超出官方声明的 `<=2.5.1`，实测可运行（见 02 文档） |
| transformers | 5.16.1 | 原生支持 `model_type=deepseek_v4`（键名映射内置于 `conversion_mapping.py`） |
| ColossalAI | 0.5.0（main, 4f9953b） | 源码安装 `--no-deps`（绕开官方钉死的版本约束） |
| peft | 0.20.0 | LoRA |
| datasets | 5.0.1 | 数据准备 |

## 踩坑记录

1. `pyext` 不兼容 Python 3.12（`inspect.getargspec` 已移除）——从镜像依赖中移除。
2. `kernels==0.16.0`（transformers finegrained-fp8 推理内核）与 NGC triton 3.3 不兼容
   （`JITFunction` 已从 `triton.runtime.autotuner` 移除）——训练走 BF16 路线后不受影响。
3. NFS 挂载写缓存延迟：宿主机新写文件在容器内可能短暂不可见；`HF_DATASETS_CACHE`
   必须放本地盘（`/tmp/hf_cache`），否则 `datasets` mmap 直接 core dump。
4. HF 下载：镜像端点（如 `hf-mirror.com`）与代理不可同时使用，二者择一。
