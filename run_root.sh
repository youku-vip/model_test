#!/usr/bin/env bash
# =============================================================================
# 以 root 运行 Isaac 0.5 容器（训练代码/数据集/模型全部挂载，不打包进镜像）
#
# 用法:
#   ./run_root.sh                    # 交互式 root shell
#   ./run_root.sh bash /opt/isaac/docker/debug_bimanual.sh   # 直接跑调试训练
#
# 环境变量（宿主机路径，按实际机器修改）:
#   IMAGE           镜像名           (默认 isaac:gpu-root)
#   CODE_DIR        源码目录         (默认 /data/isaac  -> /code)
#   DATA_DIR        数据根目录       (默认 /data        -> /data, 含 datasets/modelRepository)
#   CHECKPOINT_DIR  检查点目录       (默认 /data/checkpoints -> /checkpoint)
#   VOCAB           vocab.json 路径  (默认 /data/isaac/vocab/qwen35/vocab.json -> /vocab/vocab.json)
#   GPUS            nvidia 设备      (默认 all, 无 GPU 时可设 GPUS=none)
# =============================================================================
set -euo pipefail

IMAGE="${IMAGE:-isaac:gpu-root}"
CODE_DIR="${CODE_DIR:-/data/isaac}"
DATA_DIR="${DATA_DIR:-/data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/checkpoints}"
VOCAB="${VOCAB:-/data/isaac/vocab/qwen35/vocab.json}"
GPUS="${GPUS:-all}"

# 基本校验
for p in "$CODE_DIR" "$DATA_DIR" "$VOCAB"; do
    [ -e "$p" ] || { echo "错误: 宿主机路径不存在: $p (请设置对应环境变量)" >&2; exit 1; }
done
[ -d "$CHECKPOINT_DIR" ] || { mkdir -p "$CHECKPOINT_DIR" 2>/dev/null || echo "提示: 无法创建 $CHECKPOINT_DIR, docker 会自动创建" >&2; }

GPU_ARGS=()
if [ "$GPUS" != "none" ]; then
    GPU_ARGS=(--gpus "$GPUS")
fi

# 有 TTY(交互终端) 才加 -it，非交互(脚本/CI)时直接跑
TTY_ARGS=()
if [ -t 0 ]; then
    TTY_ARGS=(-it)
fi

echo "== 以 root 启动 $IMAGE =="
echo "   /code       <- $CODE_DIR"
echo "   /data       <- $DATA_DIR"
echo "   /checkpoint <- $CHECKPOINT_DIR"
echo "   /vocab      <- $VOCAB"
exec docker run "${TTY_ARGS[@]}" --rm "${GPU_ARGS[@]}" --shm-size 16gb \
    -v "$CODE_DIR":/code \
    -v "$DATA_DIR":/data \
    -v "$CHECKPOINT_DIR":/checkpoint \
    -v "$VOCAB":/vocab/vocab.json \
    -e QWEN35_VOCAB_PATH=/vocab/vocab.json \
    -e PYTHONPATH=/code/lerobot/src \
    "$IMAGE" "$@"
