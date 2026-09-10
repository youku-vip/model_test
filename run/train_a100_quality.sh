#!/usr/bin/env bash
set -euo pipefail

# Isaac-0.5 quality-first A100 preset.
# TRAIN_SCOPE: expert | expert_connector | expert_connector_lora
# BACKEND: fsdp1 | fsdp2 | deepspeed3

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$RUN_DIR/finetune_isaac05_advanced.py"
DATASET_DIR="${DATASET_DIR:-/data/datasets/air_fryer_wam/wam/air_fryer_pick_0810_0811_604_v30}"
BASE_PACKAGE="${BASE_PACKAGE:-/data/modelRepository/isaac0_5/isaac0_5_2026831102524/lerobot_policy}"
VOCAB="${VOCAB:-/data/algorithm/repo/vocab/qwen35/vocab.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/algorithm/repo/outputs/isaac-quality}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
SMOKE_GPUS="${SMOKE_GPUS:-0,1,2,3}"
TRAIN_SCOPE="${TRAIN_SCOPE:-expert_connector}"
BACKEND="${BACKEND:-fsdp1}"
STEPS="${STEPS:-30000}"
SAVE_FREQ="${SAVE_FREQ:-2500}"
BATCH="${BATCH:-1}"
GRAD_ACC="${GRAD_ACC:-4}"
MC="${MC:-4}"
NUM_WORKERS="${NUM_WORKERS:-12}"
CONNECTOR_LR="${CONNECTOR_LR:-1e-4}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

export QWEN35_VOCAB_PATH="$VOCAB"
export ISAAC_TRAIN_SCOPE="$TRAIN_SCOPE"
export ISAAC_CONNECTOR_LR="$CONNECTOR_LR"

COMMON=(
  --dataset-dir "$DATASET_DIR"
  --base-package "$BASE_PACKAGE"
  --vocab "$VOCAB"
  --allow-unqualified-device
  --max-seq-len 8192
  --n-obs-steps 3
  --rtc-max-delay-steps 12
  --rtc-probability 0.5
  --rtc-prefix-length 4
  --save-fp32
  --num-workers "$NUM_WORKERS"
  --video-backend torchcodec
  --train-samples-per-chunk "$MC"
  --loss-plan flow_matching_action_prediction
  --action-expert-lr 5e-5
  --max-train-steps "$STEPS"
  --connector-lr "$CONNECTOR_LR"
  --lora-r "$LORA_R"
  --lora-alpha "$LORA_ALPHA"
  --lora-dropout "$LORA_DROPOUT"
  --wandb
  --wandb-project isaac05-quality
)

# Only frozen-backbone scopes should detach VLM activations. LoRA scope needs the
# flow gradient to reach the LoRA adapters, so the advanced wrapper removes the
# detach switch when scope=expert_connector_lora.
if [ "$TRAIN_SCOPE" != "expert_connector_lora" ]; then
  COMMON+=(--detach-vlm-activations)
else
  echo "[quality] VLM LoRA enabled: VLM flow activations remain differentiable."
fi

case "${1:-full}" in
  smoke)
    python "$SCRIPT" "${COMMON[@]}" \
      --train-scope "$TRAIN_SCOPE" --backend "$BACKEND" \
      --output-dir "$OUTPUT_DIR/smoke" --gpus "$SMOKE_GPUS" \
      --steps 20 --save-freq 20 --batch-size 1 --grad-accum 8 --overwrite
    ;;
  full)
    python "$SCRIPT" "${COMMON[@]}" \
      --train-scope "$TRAIN_SCOPE" --backend "$BACKEND" \
      --output-dir "$OUTPUT_DIR" --gpus "$TRAIN_GPUS" \
      --steps "$STEPS" --save-freq "$SAVE_FREQ" \
      --batch-size "$BATCH" --grad-accum "$GRAD_ACC"
    ;;
  *)
    echo "usage: $0 [smoke|full]" >&2
    exit 2
    ;;
esac
