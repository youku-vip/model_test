#!/usr/bin/env bash
# Build the Isaac Docker images.
#
#   ./build.sh                    # GPU image  -> isaac:gpu
#   ./build.sh cpu                # CPU image  -> isaac:cpu
#   ./build.sh gpu --build-arg PRODUCTION_TORCH=1   # MK1 v12 runtime (torch 2.10.0+cu128)
#   IMAGE_NAME=my-reg/isaac:0.5 ./build.sh
set -euo pipefail
cd "$(dirname "$0")"

TARGET="${1:-gpu}"
IMAGE_NAME="${IMAGE_NAME:-isaac:${TARGET}}"
shift || true

case "$TARGET" in
    gpu) DOCKERFILE="Dockerfile" ;;
    cpu) DOCKERFILE="Dockerfile.cpu" ;;
    *) echo "usage: $0 [gpu|cpu] [docker build args...]" >&2; exit 2 ;;
esac

echo ">> Building $TARGET image as $IMAGE_NAME (dockerfile: $DOCKERFILE)"
docker build -f "$DOCKERFILE" -t "$IMAGE_NAME" "$@" .
echo ">> Done: $IMAGE_NAME"
echo ">> Run:  docker run -it --rm --gpus all --shm-size 16gb $IMAGE_NAME"
