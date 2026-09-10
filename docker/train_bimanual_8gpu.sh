#!/usr/bin/env bash
# =============================================================================
# Isaac 0.5 双臂 EEF — 正式训练脚本（8×A100-80GB, 20000 步）
#   步骤: [1] 环境自检 -> [2] 8卡正式训练 -> [3] 离线测试
# 前置: 先跑 debug_bimanual.sh 确认 20 步 loss 正常。
# =============================================================================
set -euo pipefail

# ---------- 配置（按需修改） ----------
DATASET_REPO=${DATASET_REPO:-local/bimanual}          # 数据集 repo_id
DATASET_ROOT=${DATASET_ROOT:-/data}                   # 数据集根
PACKAGE=${PACKAGE:-/data/isaac/lerobot_policy_patched} # patch 过的可写副本（先跑 debug_bimanual.sh）
OUTPUT_DIR=${OUTPUT_DIR:-/data/outputs/isaac-finetune}
VOCAB=${VOCAB:-/vocab/vocab.json}
TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3,4,5,6,7}             # 正式训练用卡
TRAIN_STEPS=${TRAIN_STEPS:-20000}
SAVE_FREQ=${SAVE_FREQ:-2500}
GRAD_ACC=${GRAD_ACC:-4}                               # 梯度累积（有效 batch = 1×4×8 = 32）

export QWEN35_VOCAB_PATH="$VOCAB"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

echo "=================================================="
echo " [正式训练] 8×A100-80GB | ${TRAIN_STEPS} 步 | 有效 batch=${GRAD_ACC}×8"
echo "=================================================="

# ---------- [1/3] 环境自检 ----------
echo "== [1/3] 环境自检 =="
isaac-check

# ---------- [2/3] 8 卡正式训练 ----------
echo "== [2/3] 8×A100 正式训练 =="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" accelerate launch \
    --num_processes=8 --mixed_precision=bf16 \
    -m lerobot.scripts.lerobot_train \
    --policy.path="$PACKAGE" \
    --dataset.repo_id="$DATASET_REPO" --dataset.root="$DATASET_ROOT" \
    --dataset.video_backend=pyav \
    --policy.train_expert_only=true \
    --batch_size=1 --gradient_accumulation_steps="$GRAD_ACC" \
    --ddp_find_unused_parameters=false \
    --steps="$TRAIN_STEPS" --save_checkpoint=true --save_freq="$SAVE_FREQ" \
    --output_dir="$OUTPUT_DIR"

# ---------- [3/3] 离线测试 ----------
echo "== [3/3] 离线测试 (预测 vs 真值) =="
python /opt/isaac/docker/eval_offline.py \
    --policy-path "$OUTPUT_DIR/checkpoints/last/pretrained_model" \
    --dataset-repo-id "$DATASET_REPO" --dataset-root "$DATASET_ROOT" \
    --n-frames 100 --episodes 3

echo "=================================================="
echo " 训练完成！检查点: $OUTPUT_DIR/checkpoints/last"
echo " 续训: accelerate launch --num_processes=8 --mixed_precision=bf16 \\"
echo "         -m lerobot.scripts.lerobot_train \\"
echo "         --config_path=$OUTPUT_DIR/checkpoints/last/train_config.json --resume=true"
echo "=================================================="
