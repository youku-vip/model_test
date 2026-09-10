#!/usr/bin/env bash
# =============================================================================
# Isaac 0.5 双臂 EEF —— A100 多卡动作头/连接层微调入口
#
# 训练目标：冻结 36B VLM/vision backbone，只训练 Isaac action expert 与其连接部分。
# A100-80GB 默认采用 8 卡 FSDP、BF16 backbone、activation detach、effective batch=32，
# 并降低 Flow MC 采样与 dataloader 压力，优先保证吞吐与显存稳定。
#
# 用法:
#   bash run_isaac05.sh smoke    # 4 卡 20 步冒烟
#   bash run_isaac05.sh full     # 8 卡正式训练，默认 20000 步
#
# 可覆盖：DATASET_DIR / BASE_PACKAGE / VOCAB / OUTPUT_DIR / TRAIN_GPUS /
#         SMOKE_GPUS / STEPS / SAVE_FREQ / BATCH / GRAD_ACC / NUM_WORKERS /
#         VIDEO_BACKEND / EXPERT_LR / SAMPLES_PER_CHUNK / TORCH_COMPILE(0/1)
# =============================================================================
set -euo pipefail

DATASET_DIR=${DATASET_DIR:-/data/datasets/air_fryer_wam/wam/air_fryer_pick_0810_0811_604_v30}
BASE_PACKAGE=${BASE_PACKAGE:-/data/modelRepository/isaac0_5/isaac0_5_2026831102524/lerobot_policy}
VOCAB=${VOCAB:-/data/algorithm/repo/vocab/qwen35/vocab.json}
OUTPUT_DIR=${OUTPUT_DIR:-/data/algorithm/repo/outputs/isaac-finetune}
RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$RUN_DIR/finetune_isaac05.py"

TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3,4,5,6,7}
SMOKE_GPUS=${SMOKE_GPUS:-0,1,2,3}
STEPS=${STEPS:-20000}
SAVE_FREQ=${SAVE_FREQ:-2500}

# A100-80GB 默认保守配置：1×4×8=32 effective batch。
BATCH=${BATCH:-1}
GRAD_ACC=${GRAD_ACC:-4}
SMOKE_GRAD_ACC=${SMOKE_GRAD_ACC:-8}
NUM_WORKERS=${NUM_WORKERS:-8}
VIDEO_BACKEND=${VIDEO_BACKEND:-torchcodec}
EXPERT_LR=${EXPERT_LR:-5e-5}
SAMPLES_PER_CHUNK=${SAMPLES_PER_CHUNK:-2}
TORCH_COMPILE=${TORCH_COMPILE:-0}

export QWEN35_VOCAB_PATH="$VOCAB"

COMMON=(
    --dataset-dir "$DATASET_DIR"
    --base-package "$BASE_PACKAGE"
    --vocab "$VOCAB"
    --allow-unqualified-device
    --max-seq-len 8192
    --n-obs-steps 3
    --detach-vlm-activations
    --rtc-max-delay-steps 12
    --rtc-probability 0.5
    --rtc-prefix-length 4
    --fsdp
    --save-fp32
    --num-workers "$NUM_WORKERS"
    --video-backend "$VIDEO_BACKEND"
    --train-samples-per-chunk "$SAMPLES_PER_CHUNK"
    --loss-plan flow_matching_action_prediction
    --action-expert-lr "$EXPERT_LR"
    --max-train-steps "$STEPS"
    --wandb
    --wandb-project isaac05-finetune
)

if [ "$TORCH_COMPILE" = "1" ]; then
    COMMON+=(--torch-compile)
    echo "[提示] torch.compile 已开启；A100/FSDP/remote-code 若出现兼容问题，设置 TORCH_COMPILE=0。"
fi

gpu_count() { awk -F, '{print NF}' <<< "$1"; }
TRAIN_GPU_COUNT=$(gpu_count "$TRAIN_GPUS")
EFFECTIVE_BATCH=$((BATCH * GRAD_ACC * TRAIN_GPU_COUNT))

if [ "$EFFECTIVE_BATCH" -ne 32 ]; then
    echo "[提示] 当前 effective batch=$EFFECTIVE_BATCH（推荐默认值 32）；如有意调整可通过 BATCH/GRAD_ACC 覆盖。"
fi

echo "=================================================="
echo " A100 多卡动作头/连接层训练 preset"
echo "=================================================="
echo " TRAIN_GPUS=$TRAIN_GPUS ($TRAIN_GPU_COUNT GPUs)"
echo " BATCH=$BATCH  GRAD_ACC=$GRAD_ACC  effective=$EFFECTIVE_BATCH"
echo " FLOW_MC=$SAMPLES_PER_CHUNK  OBS_STEPS=3  MAX_SEQ_LEN=8192"
echo " BACKBONE=Frozen/BF16 + detach  |  TRAINABLE=ActionExpert/Connector"
echo " FSDP=ON  activation_checkpointing=OFF"
echo "=================================================="

MODE=${1:-full}
case "$MODE" in
  smoke)
    echo "== [smoke] 4 卡 20 步 =="
    python "$SCRIPT" "${COMMON[@]}" \
        --gpus "$SMOKE_GPUS" --smoke --overwrite \
        --batch-size 1 --grad-accum "$SMOKE_GRAD_ACC"
    ;;
  full)
    echo "== [full] ${TRAIN_GPU_COUNT} 卡 ${STEPS} 步 =="
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

CKPT="$OUTPUT_DIR/checkpoints/last/pretrained_model"
echo "=============================================="
echo "检查点: $CKPT"
echo "部署: CUDA_VISIBLE_DEVICES=0,1 python $RUN_DIR/deploy_isaac.py --policy-path $CKPT --device-map auto --num-inference-steps 6"
echo "续训: python $SCRIPT --output-dir $OUTPUT_DIR --gpus $TRAIN_GPUS --resume --steps $STEPS --allow-unqualified-device --detach-vlm-activations"
echo "=============================================="
