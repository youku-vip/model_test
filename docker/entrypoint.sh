#!/usr/bin/env bash
# Isaac image entrypoint: ensure writable HF/torch/triton cache dirs, then exec.
set -e
mkdir -p \
    "${HF_HOME:-$HOME/.cache/huggingface}" \
    "${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}" \
    "${TORCH_HOME:-$HOME/.cache/torch}" \
    "${TRITON_CACHE_DIR:-$HOME/.cache/triton}"
exec "$@"
