# syntax=docker/dockerfile:1
#
# Isaac 0.5 — Perceptron Isaac ENVIRONMENT-ONLY image (GPU / CUDA 12.8)
# ==========================================================================
# The image contains ONLY the runtime environment (system deps, Python 3.12,
# uv, Rust, and the full Python environment in /opt/venv). Your code, model
# weights and data are MOUNTED at runtime:
#
#   docker run -it --rm --gpus all --shm-size 16gb \
#     -v /host/isaac:/code \          # 你的 isaac + lerobot 源码 (e12389c)
#     -v /host/checkpoints:/checkpoint \
#     -v /host/data:/data \
#     -e QWEN35_VOCAB_PATH=/vocab/vocab.json \
#     isaac:gpu
#
# The mounted code wins over the baked-in snapshot via PYTHONPATH=/code/src,
# so you can edit code without rebuilding the image. If /code is not mounted,
# the image's own lerobot snapshot (same commit) is used.
#
# Uses the OFFICIAL standalone-mharmony integration (lerobot e12389c):
# `uv sync` installs mharmony[qwen35]==0.1.0 from its git pin automatically
# (Rust toolchain included for the source build; no shim needed).
#
# Build args:
#   CUDA_VERSION / OS_VERSION     base CUDA image            (default 12.8.1 / 24.04)
#   PYTHON_VERSION                python version             (default 3.12)
#   LEROBOT_EXTRAS                uv extras (repeated --extra flags)
#   PRODUCTION_TORCH              1 => torch 2.10.0+cu128 MK1/H100 ABI (default 0)
#   HF_ENDPOINT                   Hugging Face endpoint, e.g. https://hf-mirror.com
#   TORCH_CUDA_ARCH_LIST          GPU archs compiled into CUDA kernels (H100 = 9.0)
#   APT_MIRROR                    Ubuntu apt mirror (HTTPS)
#   MHARMONY_GIT_URL              mharmony git base URL; rewrite for networks where
#                                 github.com is blocked (default gh-proxy).

ARG CUDA_VERSION=12.8.1
ARG OS_VERSION=24.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${OS_VERSION}

ARG PYTHON_VERSION=3.12
ARG LEROBOT_EXTRAS="--extra perceptron_isaac_cuda --extra training --extra peft"
ARG PRODUCTION_TORCH=0
ARG HF_ENDPOINT=https://huggingface.co
ARG TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"
ARG APT_MIRROR=mirrors.aliyun.com
ARG MHARMONY_GIT_URL=https://gh-proxy.com/https://github.com/perceptron-ai-inc/mharmony.git

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/venv/bin:$PATH" \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    MUJOCO_GL=egl \
    HF_ENDPOINT="${HF_ENDPOINT}" \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    VIRTUAL_ENV=/opt/venv \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# ---- 1. System dependencies + uv + Rust (as root) -------------------------
RUN sed -i \
        -e "s|http://archive.ubuntu.com/ubuntu|https://${APT_MIRROR}/ubuntu|g" \
        -e "s|http://security.ubuntu.com/ubuntu|https://${APT_MIRROR}/ubuntu|g" \
        /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates \
        libglib2.0-0 libgl1 libegl1-mesa-dev ffmpeg \
        libusb-1.0-0-dev libgeos-dev portaudio19-dev \
        cmake pkg-config ninja-build \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev \
    && curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh \
    && curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.87.0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ---- 2. Non-root user -----------------------------------------------------
RUN useradd --create-home --shell /bin/bash user_lerobot \
    && usermod -aG sudo user_lerobot

WORKDIR /lerobot
RUN chown -R user_lerobot:user_lerobot /lerobot /usr/local/cargo /usr/local/rustup \
    && mkdir -p /opt/venv /opt/isaac && chown -R user_lerobot:user_lerobot /opt

USER user_lerobot
ENV HOME=/home/user_lerobot \
    PATH="/usr/local/cargo/bin:$PATH" \
    HF_HOME=/home/user_lerobot/.cache/huggingface \
    HF_LEROBOT_HOME=/home/user_lerobot/.cache/huggingface/lerobot \
    TORCH_HOME=/home/user_lerobot/.cache/torch \
    TRITON_CACHE_DIR=/home/user_lerobot/.cache/triton

# ---- 3. Dependencies first (cached layer) --------------------------------
# causal-conv1d==1.6.0 has no PyPI wheel and its setup.py fetches a prebuilt
# wheel from github.com (blocked): pre-install the lockfile torch into the venv
# and build causal-conv1d against it (--no-build-isolation-package +
# CAUSAL_CONV1D_FORCE_BUILD=TRUE). mharmony builds from git via maturin.
COPY --chown=user_lerobot:user_lerobot \
     lerobot/pyproject.toml lerobot/uv.lock lerobot/setup.py \
     lerobot/README.md lerobot/MANIFEST.in /lerobot/

RUN sed -i "s|https://github.com/perceptron-ai-inc/mharmony.git|${MHARMONY_GIT_URL}|g" \
        pyproject.toml uv.lock \
    && uv venv /opt/venv --python python${PYTHON_VERSION} \
    && if [ "${PRODUCTION_TORCH}" = "1" ]; then \
         TORCH_PKGS="torch==2.10.0+cu128 torchvision==0.25.0+cu128"; \
         NO_TORCH="--no-install-package torch --no-install-package torchvision"; \
       else \
         TORCH_PKGS="torch==2.11.0+cu128"; \
         NO_TORCH=""; \
       fi \
    && uv pip install \
         --find-links https://download.pytorch.org/whl/cu128/torch/ \
         --find-links https://download.pytorch.org/whl/cu128/torchvision/ \
         $TORCH_PKGS setuptools wheel \
    && CAUSAL_CONV1D_FORCE_BUILD=TRUE uv sync --locked ${LEROBOT_EXTRAS} \
         --no-install-project --no-cache \
         --no-build-isolation-package causal-conv1d $NO_TORCH

# ---- 4. Project snapshot (so the image works without any mount) -----------
COPY --chown=user_lerobot:user_lerobot lerobot/ /lerobot/

RUN sed -i "s|https://github.com/perceptron-ai-inc/mharmony.git|${MHARMONY_GIT_URL}|g" \
        pyproject.toml uv.lock \
    && if [ "${PRODUCTION_TORCH}" = "1" ]; then \
         NO_TORCH="--no-install-package torch --no-install-package torchvision"; \
       else \
         NO_TORCH=""; \
       fi \
    && CAUSAL_CONV1D_FORCE_BUILD=TRUE uv sync --locked ${LEROBOT_EXTRAS} --no-cache \
         --no-build-isolation-package causal-conv1d $NO_TORCH

# Restore the production torch ABI (uv sync prunes it because it differs from the
# lockfile). --no-deps avoids the torchcodec>=2.11 metadata conflict (use pyav).
RUN if [ "${PRODUCTION_TORCH}" = "1" ]; then \
        uv pip install --no-deps \
            --find-links https://download.pytorch.org/whl/cu128/torch/ \
            --find-links https://download.pytorch.org/whl/cu128/torchvision/ \
            "torch==2.10.0+cu128" "torchvision==0.25.0+cu128"; \
    fi

# ---- 5. Helper scripts (outside mount-shadowed paths) ---------------------
COPY --chmod=0755 docker/entrypoint.sh docker/isaac-check /usr/local/bin/
COPY --chown=user_lerobot:user_lerobot docker/*.py docker/*.sh /opt/isaac/docker/

# Mounted code (isaac repo at /code) wins over the baked-in snapshot:
# PYTHONPATH=/code/lerobot/src takes precedence over site-packages.
ENV PYTHONPATH=/code/lerobot/src

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
