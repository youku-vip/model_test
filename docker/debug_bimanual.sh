#!/usr/bin/env bash
# =============================================================================
# Isaac 0.5 双臂 EEF — 调试脚本（4×A100-80GB, 20 步短跑验证）
#   步骤: [1] 环境自检 -> [2] 备份+patch 导入包 -> [3] 4卡短跑
# 运行前先跑正式训练脚本前确认 loss 正常下降、无 NaN。
# =============================================================================
set -euo pipefail

# ---------- 配置（按需修改） ----------
DATASET_DIR=${DATASET_DIR:-/data/ds}                  # 数据集实际目录（含 meta/ data/ videos/）
DATASET_REPO=${DATASET_REPO:-local/bimanual}          # 数据集 repo_id
DATASET_ROOT=${DATASET_ROOT:-/data}                   # 数据集根（按 $DATASET_ROOT/<owner>/<name> 查找）
PACKAGE=${PACKAGE:-/model/lerobot_policy}                  # 导入包（只读源，仅读取）
PATCH_DIR=${PATCH_DIR:-/data/isaac/lerobot_policy_patched} # 可写的 patch/训练副本
BACKUP="${PATCH_DIR}_backup"                               # 副本的备份目录
OUTPUT_DIR=${OUTPUT_DIR:-/data/outputs/isaac-finetune}
VOCAB=${VOCAB:-/vocab/vocab.json}
DEBUG_GPUS=${DEBUG_GPUS:-0,1,2,3}                     # 调试用卡

ACTION_DIM=20        # 你的动作维度（双臂 EEF）
PROPRIO_DIM=34       # 你的状态维度
CAMERAS="head,left,right"
IMAGE_SIZE="240,424"
FPS=10

export QWEN35_VOCAB_PATH="$VOCAB"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

echo "=================================================="
echo " [调试] ${ACTION_DIM}D 动作 / ${PROPRIO_DIM}D 状态 | 4×A100-80GB | 20 步"
echo "=================================================="

# ---------- [1/3] 环境自检 ----------
echo "== [1/3] 环境自检 =="
isaac-check
python /opt/isaac/docker/mharmony_smoke_test.py

# ---------- [2/3] 拷贝源包到可写副本 + patch ----------
echo "== [2/3] patch 可写副本 ($PATCH_DIR) =="
if [ ! -d "$PATCH_DIR" ]; then
    echo "  首次运行: 从 $PACKAGE 拷贝副本到 $PATCH_DIR"
    cp -r "$PACKAGE" "$PATCH_DIR"
else
    echo "  使用已有副本 $PATCH_DIR（如需重打，先删除它或改 PATCH_DIR）"
fi
if [ ! -d "$BACKUP" ]; then
    cp -r "$PATCH_DIR" "$BACKUP"
fi
python /opt/isaac/docker/patch_package_for_bimanual.py \
    --package "$PATCH_DIR" \
    --dataset-dir "$DATASET_DIR" \
    --action-dim "$ACTION_DIM" --proprio-dim "$PROPRIO_DIM" \
    --cameras "$CAMERAS" --image-size "$IMAGE_SIZE" --fps "$FPS"

# ---------- [3/3] 4 卡短跑验证 ----------
echo "== [3/3] 4×A100 短跑验证 (20 步) =="
CUDA_VISIBLE_DEVICES="$DEBUG_GPUS" accelerate launch \
    --num_processes=4 --mixed_precision=bf16 \
    -m lerobot.scripts.lerobot_train \
    --policy.path="$PATCH_DIR" \
    --dataset.repo_id="$DATASET_REPO" --dataset.root="$DATASET_ROOT" \
    --dataset.video_backend=pyav \
    --policy.train_expert_only=true \
    --batch_size=1 --gradient_accumulation_steps=8 \
    --ddp_find_unused_parameters=false \
    --steps=20 --save_checkpoint=true \
    --output_dir="$OUTPUT_DIR/smoke"

echo "=================================================="
echo " 调试完成！检查:"
echo "   loss 是否正常下降:   $OUTPUT_DIR/smoke"
echo "   无 NaN / 显存不爆"
echo " 确认无误后运行正式训练脚本: train_bimanual_8gpu.sh"
echo "=================================================="
