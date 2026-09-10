#!/usr/bin/env bash
# =============================================================================
# Isaac 0.5 双臂 EEF —— A100-80GB 训练编排（在【已有】生产镜像容器内直接执行）
#   [1] 4×A100-80GB 冒烟测试（20 步，验证 loss 正常、无 NaN、显存不爆）
#   [2] 8×A100-80GB 正式训练（默认 20000 步，有效 batch = 1×4×8 = 32）
#
# 前提：你已经有一个跑起来的容器（代码/数据集/模型都已挂载），本脚本只在容器内
# 运行，不需要也不建议在宿主执行 docker run。先探测真实路径：
#     python /code/run/probe_env.py
# 路径与默认值不符时用环境变量覆盖（见下）。如确实还没有容器，参考
# docker/README.md 的挂载规范用 docker run 启动（见文件末尾参考注释）。
#
# 容器内用户 user_lerobot 与宿主 uid 1000 一致，常规挂载可直接写；若只读挂载，
# 先 sudo chown -R user_lerobot:user_lerobot /code /model 或改用可写挂载。
# A100(SM80) 需要 --allow-unqualified-device：跳过便携模型的 SM90/H100 生产闸门
# （transformers/CUDA 版本检查保留；训练时冻结骨干仅 BF16 前向，与推理 parity 无关）。
# =============================================================================
set -euo pipefail

# ---------- 配置（默认值自适应容器挂载，可按需用环境变量覆盖） ----------
RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"       # <仓库>/run
REPO_ROOT_ABS="$(dirname "$RUN_DIR")"                         # <仓库>
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT_ABS/outputs/isaac-finetune"}
BASE_PACKAGE=${BASE_PACKAGE:-}                     # 留空 = 自动探测（/data/isaac/model、/model、/checkpoint）
VOCAB=${VOCAB:-/vocab/vocab.json}
SMOKE_GPUS=${SMOKE_GPUS:-0,1,2,3}                  # 冒烟测试用卡（4×A100-80GB）
TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3,4,5,6,7}          # 正式训练用卡（8×A100-80GB）
TRAIN_STEPS=${TRAIN_STEPS:-20000}
SAVE_FREQ=${SAVE_FREQ:-2500}
GRAD_ACC=${GRAD_ACC:-4}                            # 正式训练梯度累积
SMOKE_GRAD_ACC=${SMOKE_GRAD_ACC:-8}                # 冒烟测试梯度累积

# 数据集自动定位（未显式设置时，在常见挂载点里找含 meta/info.json 的目录）
if [ -z "${DATASET_DIR:-}" ]; then
    for cand in /data/local/bimanual /data/ds /data/dataset; do
        if [ -f "$cand/meta/info.json" ]; then
            DATASET_DIR=$cand
            echo "自动定位数据集: $DATASET_DIR"
            break
        fi
    done
fi
if [ -z "${DATASET_DIR:-}" ]; then
    echo "无法自动定位数据集（未找到 meta/info.json），请设置 DATASET_DIR" >&2
    echo "先用 python /code/run/probe_env.py 确认挂载路径" >&2
    exit 1
fi

# 脚本定位：/code/run 或脚本所在目录（已在配置段算出 RUN_DIR）
RUN_SCRIPT=${RUN_SCRIPT:-/code/run/finetune_isaac05.py}
if [ ! -f "$RUN_SCRIPT" ]; then
    RUN_SCRIPT="$RUN_DIR/finetune_isaac05.py"
fi

export QWEN35_VOCAB_PATH="${QWEN35_VOCAB_PATH:-$VOCAB}"
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}

# 组装公共参数（BASE_PACKAGE 留空时交给 python 自动探测）
BASE_ARGS=()
if [ -n "$BASE_PACKAGE" ]; then
    BASE_ARGS=(--base-package "$BASE_PACKAGE")
fi

# wandb 可视化（可选）：WANDB=1 时开启，并透传项目/实体/模式/服务地址
WANDB_ARGS=()
if [ "${WANDB:-0}" = "1" ]; then
    WANDB_ARGS=(--wandb)
    [ -n "${WANDB_PROJECT:-}" ] && WANDB_ARGS+=(--wandb-project "$WANDB_PROJECT")
    [ -n "${WANDB_ENTITY:-}" ] && WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
    [ -n "${WANDB_MODE:-}" ] && WANDB_ARGS+=(--wandb-mode "$WANDB_MODE")
    [ -n "${WANDB_HOST:-}" ] && WANDB_ARGS+=(--wandb-host "$WANDB_HOST")
    [ -n "${WANDB_API_KEY:-}" ] && WANDB_ARGS+=(--wandb-api-key "$WANDB_API_KEY")
    [ -n "${WANDB_RUN_ID:-}" ] && WANDB_ARGS+=(--wandb-run-id "$WANDB_RUN_ID")
fi

# ---------- [0] 环境自检 ----------
echo "=================================================="
echo " [0] 环境自检（isaac-check）"
echo "=================================================="
isaac-check || true

# ---------- [1] 4×A100 冒烟测试 ----------
echo "=================================================="
echo " [1/2] 4×A100-80GB 冒烟测试（20 步）"
echo "=================================================="
python "$RUN_SCRIPT" \
    --dataset-dir "$DATASET_DIR" \
    "${BASE_ARGS[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --gpus "$SMOKE_GPUS" \
    --grad-accum "$SMOKE_GRAD_ACC" \
    --smoke \
    --allow-unqualified-device \
    "${WANDB_ARGS[@]}"

echo "冒烟通过：检查 $OUTPUT_DIR/smoke 下 loss 正常下降、无 NaN"
echo "如异常，先排查数据/显存；确认无误后继续正式训练。"

# ---------- [2] 8×A100 正式训练 ----------
echo "=================================================="
echo " [2/2] 8×A100-80GB 正式训练（${TRAIN_STEPS} 步，有效 batch=${GRAD_ACC}×8=32）"
echo "=================================================="
python "$RUN_SCRIPT" \
    --dataset-dir "$DATASET_DIR" \
    "${BASE_ARGS[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --gpus "$TRAIN_GPUS" \
    --steps "$TRAIN_STEPS" --save-freq "$SAVE_FREQ" \
    --grad-accum "$GRAD_ACC" \
    --allow-unqualified-device \
    "${WANDB_ARGS[@]}"

echo "=================================================="
echo " 训练完成！检查点: $OUTPUT_DIR/checkpoints/last/pretrained_model"
echo " 离线评测: python /code/docker/eval_offline.py \\"
echo "    --policy-path $OUTPUT_DIR/checkpoints/last/pretrained_model \\"
echo "    --dataset-repo-id local/bimanual --dataset-root $DATASET_DIR \\"
echo "    --n-frames 100 --episodes 3"
echo " 在线部署: python /code/run/deploy_isaac.py \\"
echo "    --policy-path $OUTPUT_DIR/checkpoints/last/pretrained_model"
echo " 续训: python /code/run/finetune_isaac05.py \\"
echo "    --dataset-dir $DATASET_DIR \\"
echo "    --output-dir $OUTPUT_DIR --gpus $TRAIN_GPUS \\"
echo "    --resume --steps $TRAIN_STEPS --allow-unqualified-device"
echo "=================================================="

# -----------------------------------------------------------------------------
# 参考：如果还没有容器，用 docker/README.md 的挂载规范从宿主启动一个（仅在确实
# 没有容器时需要，已有容器时请直接在容器内执行本脚本）：
#
#   docker run -it --rm --gpus all --shm-size 16gb \
#     -v /host/isaac/repo:/code \
#     -v /host/isaac/model:/model \
#     -v /host/data:/data \
#     -v /host/vocab:/vocab:ro \
#     -e QWEN35_VOCAB_PATH=/vocab/vocab.json \
#     isaac:gpu \
#     bash /code/run/train_a100.sh
# -----------------------------------------------------------------------------
