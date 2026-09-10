#!/usr/bin/env bash
# =============================================================================
# Isaac 0.5 双臂 EEF —— 完整运行脚本（微调 + 部署提示）
#   用法:
#     bash run_isaac05.sh smoke    # 4 卡冒烟（20 步，验证跑通）
#     bash run_isaac05.sh full     # 8 卡正式训练（默认 20000 步）
#   环境变量可覆盖（常用）:
#     DATASET_DIR / BASE_PACKAGE / VOCAB / OUTPUT_DIR
#     TRAIN_GPUS / SMOKE_GPUS / STEPS / SAVE_FREQ / BATCH / GRAD_ACC
#     NUM_WORKERS / VIDEO_BACKEND / EXPERT_LR / TORCH_COMPILE(0/1)
# 说明:
#   - FSDP 分片(36B) + RTC(前缀 inpainting) + 统计量自动修复 + 保存 FP32
#   - 默认不带 --torch-compile（FSDP 下整模型编译吃显存/兼容差）；想试设 TORCH_COMPILE=1
#   - 需要 torchcodec 时 VIDEO_BACKEND=torchcodec（torch2.11 容器可试；报错换 pyav）
# =============================================================================
set -euo pipefail

# ---------- 配置（环境变量可覆盖） ----------
DATASET_DIR=${DATASET_DIR:-/data/datasets/air_fryer_wam/wam/air_fryer_pick_0810_0811_604_v30}
BASE_PACKAGE=${BASE_PACKAGE:-/data/modelRepository/isaac0_5/isaac0_5_2026831102524/lerobot_policy}
VOCAB=${VOCAB:-/data/algorithm/repo/vocab/qwen35/vocab.json}
OUTPUT_DIR=${OUTPUT_DIR:-/data/algorithm/repo/outputs/isaac-finetune}
RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$RUN_DIR/finetune_isaac05.py"

TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3,4,5,6,7}   # 正式训练 8 卡
SMOKE_GPUS=${SMOKE_GPUS:-0,1,2,3}           # 冒烟 4 卡
STEPS=${STEPS:-20000}
SAVE_FREQ=${SAVE_FREQ:-2500}
BATCH=${BATCH:-4}                           # 每卡 batch（8 卡正式有效 batch = 4×4×8=128）
GRAD_ACC=${GRAD_ACC:-4}
NUM_WORKERS=${NUM_WORKERS:-16}
VIDEO_BACKEND=${VIDEO_BACKEND:-torchcodec}  # 报解码错就换 pyav
EXPERT_LR=${EXPERT_LR:-1e-4}                # 动作头 LR（train_expert_only 实际生效的）
SAMPLES_PER_CHUNK=${SAMPLES_PER_CHUNK:-8}   # 每 chunk flow MC 采样（显存紧改 2-4）
TORCH_COMPILE=${TORCH_COMPILE:-0}

export QWEN35_VOCAB_PATH="$VOCAB"

# ---------- 公共参数 ----------
COMMON=(
    --dataset-dir "$DATASET_DIR"
    --base-package "$BASE_PACKAGE"
    --vocab "$VOCAB"
    --allow-unqualified-device
    --max-seq-len 8192 --n-obs-steps 3
    --rtc-max-delay-steps 12 --rtc-probability 0.5 --rtc-prefix-length 4
    --fsdp --save-fp32
    --num-workers "$NUM_WORKERS" --video-backend "$VIDEO_BACKEND"
    --fsdp-activation-checkpointing --train-samples-per-chunk "$SAMPLES_PER_CHUNK"
    --action-expert-lr "$EXPERT_LR" --max-train-steps "$STEPS"
    --wandb --wandb-project isaac05-finetune
)
if [ "$TORCH_COMPILE" = "1" ]; then
    COMMON+=(--torch-compile)
    echo "[提示] 已开启 --torch-compile（FSDP 下可能吃显存/兼容差，OOM 就关掉）"
fi

# ---------- 执行 ----------
MODE=${1:-full}
case "$MODE" in
  smoke)
    echo "== [smoke] 4 卡冒烟（20 步）=="
    python "$SCRIPT" "${COMMON[@]}" \
        --gpus "$SMOKE_GPUS" --smoke --overwrite \
        --batch-size 2 --grad-accum 8
    ;;
  full)
    echo "== [full] ${TRAIN_GPUS} 正式训练（${STEPS} 步，有效 batch=${BATCH}×${GRAD_ACC}×卡数）=="
    python "$SCRIPT" "${COMMON[@]}" \
        --gpus "$TRAIN_GPUS" \
        --steps "$STEPS" --save-freq "$SAVE_FREQ" \
        --batch-size "$BATCH" --grad-accum "$GRAD_ACC"
    ;;
  *)
    echo "用法: $0 [smoke|full]" >&2
    exit 2
    ;;
esac

# ---------- 结果提示 ----------
CKPT="$OUTPUT_DIR/checkpoints/last/pretrained_model"
echo "=============================================="
echo " 检查点: $CKPT"
echo
echo " 部署（2 卡，一个实例）:"
echo "  CUDA_VISIBLE_DEVICES=0,1 python $RUN_DIR/deploy_isaac.py \\"
echo "      --policy-path $CKPT --device-map auto --host 0.0.0.0 --port 8600 \\"
echo "      --num-inference-steps 6"
echo
echo " 4 卡部署 2 实例（各跨 2 卡）:"
echo "  bash $RUN_DIR/deploy_2x2.sh $CKPT"
echo
echo " 离线可视化:"
echo "  python $RUN_DIR/viz_server.py --log-dir $OUTPUT_DIR/smoke --port 8600"
echo
echo " 续训:"
echo "  python $SCRIPT --output-dir $OUTPUT_DIR --gpus $TRAIN_GPUS \\"
echo "      --resume --steps $STEPS --allow-unqualified-device"
echo "=============================================="
