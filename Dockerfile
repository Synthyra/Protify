# syntax=docker/dockerfile:1.7
# FastPLMs 1.0 validates PyTorch 2.13 against CUDA 13.0. The devel image also
# supplies ptxas for compiled attention kernels, which PyTorch no longer bundles.
FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04@sha256:8b2705ea7a8653ad3451b46ab835eced92d77b44e671b9cf3ad4f95fbb2efe5e

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/app \
    PATH=/opt/venv/bin:/usr/local/bin:$PATH \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    TOKENIZERS_PARALLELISM=true \
    PROJECT_ROOT=/workspace \
    HF_XET_HIGH_PERFORMANCE=1 \
    DISABLE_PANDERA_IMPORT_WARNING=True \
    HF_HOME=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch \
    XDG_CACHE_HOME=/workspace/.cache \
    WANDB_DIR=/workspace/logs \
    TQDM_CACHE=/workspace/.cache/tqdm

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential curl git ca-certificates \
        python3.12 python3.12-dev python3.12-venv \
        ninja-build && \
    python3.12 -m venv /opt/venv && \
    ln -sf /opt/venv/bin/python /usr/local/bin/python && \
    ln -sf /opt/venv/bin/pip /usr/local/bin/pip

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip setuptools
RUN pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
RUN pip install -r requirements.txt

COPY . .

WORKDIR /workspace

CMD ["bash"]
