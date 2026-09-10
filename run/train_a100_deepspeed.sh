#!/usr/bin/env bash
set -euo pipefail

# Isaac-0.5 / A100-80GB / DeepSpeed ZeRO-3 preset.
# ZeRO-2 is intentionally not used: 36B BF16 parameters are ~72GB before
# runtime overhead, while ZeRO-2 leaves parameters replicated on every GPU.

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
DATASET_DIR="${DATASET_DIR:-/data/local/bimanual}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/outputs/isaac-finetune-ds3}"
STEPS="${STEPS:-20000}"
SAVE_FREQ="${SAVE_FREQ:-2500}"
BATCH="${BATCH:-1}"
GRAD_ACC="${GRAD_ACC:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAMPLES_PER_CHUNK="${SAMPLES_PER_CHUNK:-2}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchcodec}"
EXPERT_LR="${EXPERT_LR:-5e-5}"
DS_CONFIG="${DS_CONFIG:-/code/run/deepspeed_zero3_a100.json}"

export ISAAC_DEEPSPEED_CONFIG="$DS_CONFIG"

python /code/run/finetune_isaac05_deepspeed.py \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --steps "$STEPS" \
  --save-freq "$SAVE_FREQ" \
  --gpus "$GPUS" \
  --batch-size "$BATCH" \
  --grad-accum "$GRAD_ACC" \
  --num-workers "$NUM_WORKERS" \
  --video-backend "$VIDEO_BACKEND" \
  --train-samples-per-chunk "$SAMPLES_PER_CHUNK" \
  --action-expert-lr "$EXPERT_LR" \
  --max-seq-len 8192 \
  --n-obs-steps 3 \
  --detach-vlm-activations \
  --rtc-max-delay-steps 12 \
  --rtc-probability 0.5 \
  --rtc-prefix-length 4 \
  --loss-plan flow_matching_action_prediction \
  --train-storage-fp32 false \
  --save-fp32 \
  --allow-unqualified-device \
  --wandb \
  --wandb-project isaac05-finetune-ds3
