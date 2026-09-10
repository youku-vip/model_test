#!/usr/bin/env bash
# =============================================================================
# 本地 wandb 可视化面板 —— 默认端口 8600
#   用法: bash wandb_server.sh [port]       # 默认 8600
#   启动后训练侧加: --wandb --wandb-host http://<host>:8600 --wandb-api-key <key>
#   （wandb 本地服务需要 docker 来跑后端；无 docker 时改用云端:
#      export WANDB_API_KEY=<key> 然后 --wandb 即可，打开日志里 wandb 打印的 URL）
# =============================================================================
set -euo pipefail
PORT=${1:-8600}
HOST=${WANDB_SERVER_HOST:-0.0.0.0}

echo "启动本地 wandb 服务: http://${HOST}:${PORT}  （首次会初始化本地后端，需 docker）"
exec wandb server --host "${HOST}" --port "${PORT}"
