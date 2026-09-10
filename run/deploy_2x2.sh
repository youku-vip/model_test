#!/usr/bin/env bash
# =============================================================================
# 4 卡部署 2 个模型实例，每个实例跨 2 卡（HF device_map=auto 模型并行）
#   实例1: GPU 0,1 -> 端口 8600
#   实例2: GPU 2,3 -> 端口 8601
# 用法:
#   bash deploy_2x2.sh <policy-path> [--num-inference-steps 6] [--torch-compile]
#   （可选 --num-inference-steps 4-6 大幅降推理延迟；--torch-compile 再加速但有兼容风险）
# =============================================================================
set -euo pipefail

POLICY=${1:?用法: deploy_2x2.sh <policy-path> [额外参数...]}
shift || true
EXTRA=("$@")

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOCAB=${VOCAB:-/data/algorithm/repo/vocab/qwen35/vocab.json}
export QWEN35_VOCAB_PATH="$VOCAB"

echo "启动 2 个部署实例（各跨 2 卡）："
echo "  实例1: GPU 0,1 -> :8600"
echo "  实例2: GPU 2,3 -> :8601"

# 实例1
CUDA_VISIBLE_DEVICES=0,1 python "$RUN_DIR/deploy_isaac.py" \
    --policy-path "$POLICY" --device-map auto --host 0.0.0.0 --port 8600 \
    "${EXTRA[@]}" &
PID1=$!

# 实例2
CUDA_VISIBLE_DEVICES=2,3 python "$RUN_DIR/deploy_isaac.py" \
    --policy-path "$POLICY" --device-map auto --host 0.0.0.0 --port 8601 \
    "${EXTRA[@]}" &
PID2=$!

trap 'echo "停止实例"; kill $PID1 $PID2 2>/dev/null || true' INT TERM
wait
