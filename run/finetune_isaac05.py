#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Isaac 0.5 双臂 EEF —— 流匹配(Flow Matching)微调脚本（LeRobot v3.0 数据集）。

设计目标
========
以 /data/isaac/model 下的 Isaac-0.5 便携包（PerceptronIsaacPolicy + 原生
mharmony 渲染）为底座，用一份 LeRobot v3.0 双臂数据集（例如 3 相机 head/left/right、
34D observation.state、20D action、10fps）做 **action-expert-only 流匹配微调**
（``train_expert_only=true``，冻结 36B null-MoE VLM + vision，只训练 MolmoAct
动作专家头；loss_plan 保持 ``text_ntp_fast_flow_action``，即专家输出走 flow matching
损失）。这正好绕开 Isaac-0.5 全量训练需要 FP32 存储而 grouped_mm 专家要求 BF16
的冲突 —— 见 ``docker/patches/isaac05-expert-only-training.patch``。

脚本做的事（可独立复跑，幂等）：
  1. 校验数据集 meta（v3.0、维度、相机、fps、tasks、stats）；
  2. 从 ``data/chunk-*/file-*.parquet``（嵌套 chunk 布局）读取全量 action /
     observation.state，重算每维 q01/q99（写回 patched 包的 isaac_stats.json，
     供训练/部署共用；若数据集缺 meta/stats.json 则一并生成）；
  3. 复制基础包到可写副本，并 patch config.json + policy_preprocessor.json
     + policy_postprocessor.json（相机/尺寸/维度/fps/特征名 —— 只改 config.json
     不够，序列化 processor step 的几何字段也必须一致，否则 pack step 渲染崩溃）；
  4. 检查 lerobot 源码是否已应用 expert-only 训练补丁，缺失则自动 git apply；
  5. 用 accelerate 拉起多卡 ``lerobot_train``（--policy.train_expert_only=true）。

用法（在 isaac:gpu 镜像内、repo 挂载到 /code、数据与权重挂载就绪后）：
    # 20 步冒烟验证（等价 docker/debug_bimanual.sh）
    python /code/run/finetune_isaac05.py --dataset-dir /data/local/bimanual --smoke

    # 正式训练：8×GPU 20000 步，有效 batch = 1×4×8 = 32
    python /code/run/finetune_isaac05.py \
        --dataset-dir /data/local/bimanual \
        --output-dir /data/outputs/isaac-finetune \
        --steps 20000 --save-freq 2500 --gpus 0,1,2,3,4,5,6,7

    # 训练产物: <output>/checkpoints/last/pretrained_model
    # 离线评测: python /code/docker/eval_offline.py --policy-path <产物> ...
    # 在线部署: python /code/run/deploy_isaac.py --policy-path <产物> ...

说明：
  * 硬件：便携包远程代码 modeling_isaac05.py 的 require_qualified_runtime() 强制
    transformers==5.5.4 / CUDA 12.8 / SM90(H100)，请用生产镜像
    （PRODUCTION_TORCH=1）并在 H100 上训练。A100(SM80) 训练加
    --allow-unqualified-device：脚本向模型写入幂等豁免
    （ISAAC_ALLOW_UNQUALIFIED_DEVICE=1 时跳过 capability/设备名检查，
    transformers/CUDA 版本检查保留）。4 卡冒烟、8 卡正式见 repo/run/train_a100.sh。
  * 数据不足：动作目标按 chunk_size=50 帧采样，短于 50 帧的 episode 会被采样器
    丢弃（脚本会打印提示）。
  * 数据集位置：--dataset-dir 直接指向含 meta/、data/、videos/ 的目录。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 仓库 / 源码 / 补丁定位（repo/run 下运行；容器内即 /code/run）
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_SRC = REPO_ROOT / "lerobot" / "src" / "lerobot"
PATCH_FILE = REPO_ROOT / "docker" / "patches" / "isaac05-expert-only-training.patch"
TRAIN_GATE_FILE = (
    LEROBOT_SRC / "policies" / "perceptron_isaac" / "modeling_perceptron_isaac.py"
)
TRAIN_GATE_MARKER = "# [PATCH] Expert-only fine-tuning"  # 补丁内注释标记

# 便携模型自带 SM90/H100 生产闸门（modeling_isaac05.py::require_qualified_runtime）。
# A100(SM80) 训练需要 opt-in 豁免：env ISAAC_ALLOW_UNQUALIFIED_DEVICE=1 时跳过
# capability/设备名检查（transformers/CUDA 版本检查保留，仍需生产镜像）。
MODEL_GATE_FILE_NAME = "modeling_isaac05.py"
MODEL_GATE_MARKER = "# [PATCH] A100 expert-only fine-tuning"
MODEL_GATE_ENV = "ISAAC_ALLOW_UNQUALIFIED_DEVICE"
# 在 CUDA 版本检查之后、capability 检查之前插入豁免
MODEL_GATE_INSERT_ANCHOR = (
    '    if torch.version.cuda != _QUALIFIED_CUDA_VERSION:\n'
    '        installed_cuda = torch.version.cuda or "unavailable"\n'
    '        raise RuntimeError(\n'
    '            f"Isaac-0.5 CUDA inference requires CUDA {_QUALIFIED_CUDA_VERSION}, got {installed_cuda}."\n'
    '        )\n'
    '    capability = torch.cuda.get_device_capability(resolved_device)\n'
)
MODEL_GATE_INSERT_REPLACEMENT = (
    '    if torch.version.cuda != _QUALIFIED_CUDA_VERSION:\n'
    '        installed_cuda = torch.version.cuda or "unavailable"\n'
    '        raise RuntimeError(\n'
    '            f"Isaac-0.5 CUDA inference requires CUDA {_QUALIFIED_CUDA_VERSION}, got {installed_cuda}."\n'
    '        )\n'
    f'    if os.environ.get("{MODEL_GATE_ENV}") == "1":\n'
    f'        # {MODEL_GATE_MARKER}: 冻结骨干 BF16 前向即可训练，跳过 SM90/H100 生产闸门\n'
    '        return\n'
    '    capability = torch.cuda.get_device_capability(resolved_device)\n'
)
# 补齐 modeling_isaac05.py 顶部缺失的 import os
MODEL_GATE_IMPORT_ANCHOR = "import torch\nimport torch.nn.functional as F\n"
MODEL_GATE_IMPORT_REPLACEMENT = "import os\nimport torch\nimport torch.nn.functional as F\n"

STATS_FEATURES = ("action", "observation.state")

# 容器内常见挂载布局下自动探测基础包/词表（宿主路径风格与容器路径风格都试）
BASE_PACKAGE_CANDIDATES = (
    "/data/isaac/model/lerobot_policy",
    "/model/lerobot_policy",
    "/checkpoint/lerobot_policy",
    "/checkpoints/lerobot_policy",
)
VOCAB_CANDIDATES = (
    "/vocab/vocab.json",
    "/data/isaac/vocab/qwen35/vocab.json",
)


def detect_base_package() -> Path:
    """在候选挂载点里找导入的 Isaac-0.5 LeRobot 包（含 config.json 且 type=perceptron_isaac）。"""
    for candidate in BASE_PACKAGE_CANDIDATES:
        path = Path(candidate)
        if not (path / "config.json").is_file():
            continue
        try:
            cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(cfg, dict) and cfg.get("type") == "perceptron_isaac":
            return path
    raise FileNotFoundError(
        "自动探测不到 Isaac-0.5 基础包，请用 --base-package 指定（候选: "
        + ", ".join(BASE_PACKAGE_CANDIDATES)
        + "）"
    )


def detect_vocab() -> str | None:
    for candidate in VOCAB_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    return None


def require_writable(path: Path, hint: str) -> None:
    """容器内权限预检：自动 patch 需要写挂载目录；失败给出容器内可执行的修复提示。"""
    if os.access(path, os.W_OK):
        return
    raise PermissionError(
        f"{path} 不可写。{hint}\n"
        "容器内修复：sudo chown -R user_lerobot:user_lerobot <挂载路径>  "
        "或挂载时去掉 :ro 改为可写（如 -v /host/model:/model）"
    )


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def _flatten_names(names) -> list[str] | None:
    """meta features 的 names 可能是 None / 平铺 list / {"motors": [...]}。"""
    if names is None:
        return None
    if isinstance(names, dict):
        out: list[str] = []
        for value in names.values():
            if isinstance(value, (list, tuple)):
                out.extend(str(v) for v in value)
            else:
                out.append(str(value))
        return out or None
    if isinstance(names, (list, tuple)):
        return [str(v) for v in names]
    return [str(names)]


def load_dataset_info(dataset_dir: Path) -> dict:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"缺少数据集 meta: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def iter_data_files(dataset_dir: Path) -> list[Path]:
    """LeRobot v3.0 嵌套布局: data/chunk-{index:03d}/file-{index:03d}.parquet。"""
    files = sorted((dataset_dir / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"{dataset_dir / 'data'} 下没有 chunk-*/file-*.parquet")
    return files


def validate_dataset(
    dataset_dir: Path,
    info: dict,
    *,
    action_dim: int,
    proprio_dim: int,
    cameras: list[str],
    fps: float,
) -> None:
    if str(info.get("codebase_version")) != "v3.0":
        raise ValueError(f"数据集 codebase_version 必须为 v3.0，得到 {info.get('codebase_version')!r}")

    features = info.get("features") or {}
    state_feature = features.get("observation.state")
    action_feature = features.get("action")
    if state_feature is None or action_feature is None:
        raise ValueError("数据集 features 缺少 observation.state 或 action")

    state_dim = int(state_feature["shape"][0])
    action_dim_actual = int(action_feature["shape"][0])
    if state_dim != proprio_dim:
        raise ValueError(f"observation.state 维度 {state_dim} != --proprio-dim {proprio_dim}")
    if action_dim_actual != action_dim:
        raise ValueError(f"action 维度 {action_dim_actual} != --action-dim {action_dim}")

    missing = [cam for cam in cameras if f"observation.images.{cam}" not in features]
    if missing:
        raise ValueError(f"数据集缺少相机特征: {missing}")

    declared_fps = float(info.get("fps"))
    if abs(declared_fps - float(fps)) > 1e-6:
        raise ValueError(f"数据集 fps={declared_fps} != --fps {fps}")

    total_tasks = int(info.get("total_tasks", 0))
    if total_tasks < 1 or not (dataset_dir / "meta" / "tasks.parquet").is_file():
        raise FileNotFoundError("任务条件训练需要 meta/tasks.parquet（total_tasks>=1）")

    episodes_file = dataset_dir / "meta" / "episodes"
    if not any(episodes_file.glob("chunk-*/file-*.parquet")):
        raise FileNotFoundError(f"缺少 episode 元数据: {episodes_file}/chunk-*/file-*.parquet")

    print(
        f"数据集 OK: v3.0 | state={state_dim}D action={action_dim}D "
        f"cameras={cameras} fps={declared_fps} episodes={info.get('total_episodes')} "
        f"tasks={total_tasks}"
    )


def warn_short_episodes(dataset_dir: Path, chunk_size: int) -> None:
    """短于动作目标的 episode 在采样时会被丢弃，给出提示。"""
    try:
        ep_files = sorted((dataset_dir / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        lengths = []
        for pq in ep_files:
            df = pd.read_parquet(pq, columns=["length"])
            lengths.extend(int(v) for v in df["length"].tolist())
        if not lengths:
            return
        short = sum(1 for length in lengths if length < chunk_size)
        if short:
            print(
                f"[提示] {short}/{len(lengths)} 个 episode 短于 chunk_size={chunk_size}，"
                "采样器会丢弃这些 episode 的样本（动作目标需要 chunk_size 帧）。"
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[提示] 无法统计 episode 长度（{exc}），跳过。")


# ---------------------------------------------------------------------------
# 统计量计算
# ---------------------------------------------------------------------------
def load_vector_columns(dataset_dir: Path, columns: list[str]) -> dict[str, np.ndarray]:
    """读取所有 parquet 帧，按列拼接成 [N, D] float32（嵌套 chunk 布局）。"""
    acc: dict[str, list[np.ndarray]] = {col: [] for col in columns}
    total = 0
    for pq in iter_data_files(dataset_dir):
        df = pd.read_parquet(pq, columns=columns)
        for col in columns:
            arr = np.asarray(df[col].tolist(), dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            elif arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
            acc[col].append(arr)
        total += len(df)
    result = {col: np.concatenate(acc[col], axis=0) for col in columns}
    print(f"读取统计样本: {total} 帧 -> action={result['action'].shape} state={result['observation.state'].shape}")
    return result


def compute_stats(arr: np.ndarray) -> dict[str, list[float]]:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"统计输入必须为 [N, D]，得到 {arr.shape}")
    with np.errstate(all="ignore"):
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        minimum = np.nanmin(arr, axis=0)
        maximum = np.nanmax(arr, axis=0)
        q01 = np.nanpercentile(arr, 1.0, axis=0)
        q99 = np.nanpercentile(arr, 99.0, axis=0)
    stack = np.stack([mean, std, minimum, maximum, q01, q99])
    if not np.isfinite(stack).all():
        bad = np.argwhere(~np.isfinite(stack))
        raise ValueError(f"统计量含 NaN/Inf（前 {len(bad)} 处），请检查数据是否有全 NaN 的维度")
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "min": minimum.tolist(),
        "max": maximum.tolist(),
        "q01": q01.tolist(),
        "q99": q99.tolist(),
    }


def write_isaac_stats(
    package: Path,
    *,
    action_arr: np.ndarray,
    state_arr: np.ndarray,
    chunk_size: int,
    fps: float,
    action_dim: int,
    proprio_dim: int,
) -> None:
    action = compute_stats(action_arr)
    state = compute_stats(state_arr)
    stats = {
        "action": {"q01": action["q01"], "q99": action["q99"],
                   "min": action["min"], "max": action["max"]},
        "action_dim": int(action_dim),
        "action_horizon": int(chunk_size),
        "action_normalization_eps": 1e-06,
        "action_representation": "absolute",
        "clip_normalized_actions": True,
        "clip_normalized_max": 10.0,
        "profile_id": None,
        "profile_scope": None,
        "proprio": {"q01": state["q01"], "q99": state["q99"],
                    "min": state["min"], "max": state["max"]},
        "proprio_dim": int(proprio_dim),
        "proprio_normalization_eps": 1e-06,
        "relative_exclude_joints": [],
        "schema": "flow_matching_stats_v1",
        "state_action_schema": f"custom_bimanual_eef_{action_dim}",
        "stats_sha256": None,
        "target_fps": float(fps),
        "validation_status": None,
    }
    digest = hashlib.sha256(
        json.dumps(
            {k: v for k, v in stats.items() if k != "stats_sha256"},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    stats["stats_sha256"] = digest
    out = package / "isaac_stats.json"
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"已写 {out}（action={action_dim}D, proprio={proprio_dim}D, chunk={chunk_size}, fps={fps}）")


def ensure_dataset_stats_json(dataset_dir: Path, arrays: dict[str, np.ndarray]) -> None:
    """保证 meta/stats.json 存在且含 action / observation.state 的 q01/q99。

    训练时 make_pre_post_processors 会把 dataset.meta.stats 转成 Isaac 归一化
    统计（_dataset_stats_to_isaac_stats 要求 q01/q99）。若数据集本身已带 stats.json
    且字段齐全则不动它；否则只补齐缺失的两个向量特征。
    """
    stats_path = dataset_dir / "meta" / "stats.json"
    existing: dict = {}
    if stats_path.is_file():
        existing = json.loads(stats_path.read_text(encoding="utf-8"))
    changed = False
    for feature in STATS_FEATURES:
        # 始终用 parquet 实际重算的统计量覆盖这两个向量特征（含 min/max），
        # 避免数据集自带 stats.json 退化（q01==q99 折叠）导致归一化爆炸 -> 全样本 outlier。
        fresh = compute_stats(arrays[feature])
        block = existing.get(feature)
        if isinstance(block, dict) and block.get("q01") == fresh["q01"] and block.get("q99") == fresh["q99"] \
                and block.get("min") == fresh["min"] and block.get("max") == fresh["max"]:
            continue
        print(f"[stats.json] 用实际数据重算 {feature} 统计量（保证 q01/q99/min/max 一致）")
        existing[feature] = fresh
        changed = True
    if changed:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        print(f"已写 {stats_path}")


# ---------------------------------------------------------------------------
# 包 patch（config.json + 序列化 processor 几何）
# ---------------------------------------------------------------------------
def _patch_pack_step_json(json_path: Path, updates: dict) -> bool:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    changed = False
    for step in data.get("steps", []):
        name = step.get("registry_name")
        if name == "perceptron_isaac_mharmony_pack":
            cfg = step["config"]
            for key, value in updates.items():
                if cfg.get(key) != value:
                    cfg[key] = value
                    changed = True
    if changed:
        json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return changed


def _patch_unnormalize_step_json(json_path: Path, updates: dict) -> bool:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    changed = False
    for step in data.get("steps", []):
        if step.get("registry_name") == "perceptron_isaac_action_unnormalize":
            cfg = step["config"]
            for key, value in updates.items():
                if cfg.get(key) != value:
                    cfg[key] = value
                    changed = True
    if changed:
        json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return changed


def patch_package(
    package: Path,
    *,
    action_dim: int,
    proprio_dim: int,
    cameras: list[str],
    image_size: list[int],
    fps: float,
    action_names: list[str] | None,
    state_names: list[str] | None,
    n_obs_steps: int | None = None,
    rtc_max_delay_steps: int = 0,
    rtc_probability: float | None = None,
    rtc_delay_sampling: str = "uniform",
    rtc_poisson_mean: float = 5.0,
    rtc_prefix_length: int = 0,
) -> None:
    """patch 可写副本：config.json 与序列化 processor 几何必须一致。

    注意：pack step（policy_preprocessor.json）的 camera_order/image_keys/
    image_size/action_dim/proprio_dim/target_fps 是渲染与打包的权威几何；只改
    config.json 会导致 mharmony pack step 仍按 7D/8D、双相机渲染而崩溃。
    """
    cfg_path = package / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg_updates = {
        "action_dim": int(action_dim),
        "proprio_dim": int(proprio_dim),
        "camera_order": list(cameras),
        "image_size": [int(x) for x in image_size],
        "target_fps": float(fps),
        "action_feature_names": action_names,
        "state_feature_names": state_names,
        "robot_type": "custom_bimanual",
        "control_mode": "ee",
        # 本地训练不推 Hub：基础包 push_to_hub=true 且无 repo_id，validate 会 raise。
        "push_to_hub": False,
        # 自定义双臂 EEF 不在 bi_yam/so100_so101/libero 白名单内：关闭严格部署契约
        # 校验（_validate_known_deployment_contract），否则 __post_init__ 直接 raise。
        "strict_hardware_feature_contract": False,
        "strict_environment_feature_contract": False,
        # 便携 isaac_0_5 包按 trained_steps=1 合成工件身份，保持与契约一致。
        "trained_steps": 1,
    }
    if n_obs_steps is not None:
        cfg_updates["n_obs_steps"] = int(n_obs_steps)
    # RTC（基础模型动作专家本就 rtc_max_delay_steps=12 / probability=0.5，微调延续）
    cfg_updates["flow_rtc_max_delay_steps"] = int(rtc_max_delay_steps)
    cfg_updates["rtc_probability"] = None if rtc_probability is None else float(rtc_probability)
    cfg_updates["flow_rtc_delay_sampling"] = str(rtc_delay_sampling)
    cfg_updates["rtc_poisson_mean"] = float(rtc_poisson_mean)
    cfg_updates["rtc_prefix_length"] = int(rtc_prefix_length)
    cfg.update(cfg_updates)
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"patched {cfg_path}")

    pack_updates = {
        "camera_order": list(cameras),
        "image_size": [int(x) for x in image_size],
        "image_keys": [f"observation.images.{cam}" for cam in cameras],
        "action_dim": int(action_dim),
        "proprio_dim": int(proprio_dim),
        "target_fps": float(fps),
        "robot_type": "custom_bimanual",
        "action_feature_names": action_names,
        "state_feature_names": state_names,
    }
    if n_obs_steps is not None:
        pack_updates["n_obs_steps"] = int(n_obs_steps)
    pre = package / "policy_preprocessor.json"
    if pre.is_file() and _patch_pack_step_json(pre, pack_updates):
        print(f"patched {pre}（pack step 几何 -> {cameras} / {image_size} / {action_dim}D / {proprio_dim}D）")

    unnorm_updates = {
        "action_feature_names": action_names,
        "state_feature_names": state_names,
    }
    post = package / "policy_postprocessor.json"
    if post.is_file() and _patch_unnormalize_step_json(post, unnorm_updates):
        print(f"patched {post}（unnormalize step 特征名）")


def prepare_package(base_package: Path, patch_dir: Path) -> Path:
    """复制基础包到可写副本（已存在则复用）。

    可写副本必须与模型目录同级（默认 <model_dir>/lerobot_policy_patched），
    这样 config.json 里的 hf_model_path=".." 仍能按便携 isaac_0_5 布局解析到
    模型根目录（_resolve_checkpoint_local_paths 只接受这种 sibling 布局）。
    """
    if not base_package.is_dir():
        raise FileNotFoundError(f"基础包不存在: {base_package}")
    if patch_dir.exists():
        print(f"复用可写副本 {patch_dir}（如需重打，请删除后重跑）")
        return patch_dir
    patch_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(base_package, patch_dir)
    print(f"已从 {base_package} 复制可写副本 -> {patch_dir}")
    return patch_dir


# ---------------------------------------------------------------------------
# lerobot 源码补丁（expert-only 训练闸门）
# ---------------------------------------------------------------------------
def _convert_weights_to_fp32(policy_dir: Path) -> None:
    """把 pretrained_model 里的模型权重（model.safetensors / model-*.safetensors）原地转成 FP32。

    训练用 BF16 存储省显存时，用这个保证落盘的权重是 FP32（满足"保存使用 FP32"）。
    """
    from safetensors.torch import load_file, save_file  # 容器内可用

    policy_dir = Path(policy_dir)
    if not policy_dir.is_dir():
        raise FileNotFoundError(f"检查点目录不存在: {policy_dir}")
    single = policy_dir / "model.safetensors"
    files: list[Path] = []
    if single.is_file():
        files.append(single)
    files.extend(sorted(policy_dir.glob("model-*.safetensors")))
    if not files:
        raise RuntimeError(f"{policy_dir} 下没有 model*.safetensors，无法转 FP32")
    total = 0
    for f in files:
        state = load_file(str(f))
        new_state = {
            k: (v.float() if v.is_floating_point() else v) for k, v in state.items()
        }
        save_file(new_state, str(f))
        total += len(new_state)
    print(f"[save-fp32] 已把 {policy_dir} 的 {len(files)} 个权重分片转成 FP32（{total} 个张量）")


def _git_apply_patch(patch_file: Path, tree_root: Path) -> None:
    """git apply 补丁（cwd=补丁相对路径的仓库根；非 git 目录也可用）。"""
    result = subprocess.run(
        ["git", "apply", str(patch_file)],
        cwd=str(tree_root),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"在 {tree_root} 应用补丁失败：\n{result.stderr}")


def _runtime_lerobot_src() -> Path | None:
    """运行时实际 import 的 lerobot 源码根（.../src/lerobot），失败返回 None。"""
    try:
        import lerobot  # noqa: F401

        return Path(lerobot.__file__).resolve().parent
    except Exception:  # noqa: BLE001
        return None


def _gate_exempt_present(gate_file: Path) -> bool:
    """训练闸门是否已放开：_require_native_training_supported 带 train_expert_only 豁免。"""
    text = gate_file.read_text(encoding="utf-8")
    if TRAIN_GATE_MARKER in text:
        return True
    # 功能检查：_require_native_training_supported 的 if 里带 train_expert_only 豁免
    return 'not bool(getattr(self.config, "train_expert_only", False))' in text or (
        "train_expert_only" in text and "allow_unqualified_training_device" in text
    )


GATE_TEXT_ANCHOR = (
    "        if self.config.hf_model_path and self._config_declares_mk1(self.config):\n"
    "            raise RuntimeError(\n"
)
GATE_TEXT_REPLACEMENT = (
    "        if (\n"
    "            self.config.hf_model_path\n"
    "            and self._config_declares_mk1(self.config)\n"
    '            and not bool(getattr(self.config, "train_expert_only", False))\n'
    "        ):\n"
    "            raise RuntimeError(\n"
)
GATE_FUNC_ANCHOR = "    def _require_native_training_supported(self) -> None:\n"


def _apply_train_gate_text_patch(gate_file: Path) -> None:
    """精确文本补丁：给 _require_native_training_supported 加 train_expert_only 豁免。

    不依赖整包 git apply（mk1 文件版本差异可能导致 apply 失败），便携训练路径只需这个闸门。
    """
    text = gate_file.read_text(encoding="utf-8")
    if GATE_TEXT_ANCHOR not in text:
        raise RuntimeError(
            f"{gate_file} 的训练闸门结构与预期不符，请手工处理（_require_native_training_supported）"
        )
    text = text.replace(GATE_TEXT_ANCHOR, GATE_TEXT_REPLACEMENT, 1)
    if GATE_FUNC_ANCHOR in text:
        text = text.replace(
            GATE_FUNC_ANCHOR,
            GATE_FUNC_ANCHOR + f"        # {TRAIN_GATE_MARKER}\n",
            1,
        )
    gate_file.write_text(text, encoding="utf-8")
    if not _gate_exempt_present(gate_file):
        raise RuntimeError("训练闸门文本补丁应用后校验失败")
    print(f"[训练闸门] {gate_file} 已加 train_expert_only 豁免")


def ensure_expert_only_training_patch(*, apply: bool) -> None:
    """确保 modeling_perceptron_isaac.py 的 _require_native_training_supported
    带 train_expert_only 豁免（否则 MK1/便携包训练直接 raise）。

    优先检测功能闸门是否已放开；未放开时先尝试整包 git apply，失败则回退到
    精确文本补丁（只打这个闸门，mk1 文件差异忽略）。同时做运行时一致性兜底。
    """
    if not TRAIN_GATE_FILE.is_file():
        raise FileNotFoundError(f"找不到 lerobot 源码: {TRAIN_GATE_FILE}")
    if _gate_exempt_present(TRAIN_GATE_FILE):
        print("lerobot 训练闸门已放开（train_expert_only 豁免在），跳过")
    else:
        if not apply:
            raise RuntimeError(
                "lerobot 源码训练闸门未放开，且指定了 --skip-patch-src。请先应用补丁："
                f"git -C {REPO_ROOT / 'lerobot'} apply {PATCH_FILE}"
            )
        require_writable(
            TRAIN_GATE_FILE.parent,
            "自动应用补丁需要写挂载的 lerobot 源码目录。",
        )
        if PATCH_FILE.is_file():
            try:
                _git_apply_patch(PATCH_FILE, REPO_ROOT / "lerobot")
                print(f"已应用 expert-only 训练补丁: {PATCH_FILE.name}")
            except RuntimeError as exc:
                print(f"[警告] 整包 git apply 失败（版本差异）：{exc}")
                print("        回退：仅对训练闸门做精确文本补丁（便携训练路径只需这个）")
        if not _gate_exempt_present(TRAIN_GATE_FILE):
            _apply_train_gate_text_patch(TRAIN_GATE_FILE)
        if not _gate_exempt_present(TRAIN_GATE_FILE):
            raise RuntimeError("训练闸门仍未放开，无法继续")

    # ---- 运行时一致性兜底 ----
    runtime_src = _runtime_lerobot_src()
    mounted_src = LEROBOT_SRC.resolve()
    if runtime_src is None or runtime_src == mounted_src:
        return
    runtime_gate = runtime_src / "policies" / "perceptron_isaac" / "modeling_perceptron_isaac.py"
    if not runtime_gate.is_file():
        print(f"[警告] 运行时 lerobot 在 {runtime_src}，但找不到该路径下的训练闸门文件，无法兜底")
        return
    if _gate_exempt_present(runtime_gate):
        print(f"运行时 lerobot（{runtime_src}）训练闸门也已放开")
        return
    print(
        f"[警告] 运行时 import 的 lerobot 是 {runtime_src}（不是挂载仓库 {mounted_src}）；"
        "自动对运行时源码打训练闸门文本补丁。"
        "更推荐：export PYTHONPATH=<仓库>/lerobot/src 让挂载仓库优先。"
    )
    try:
        require_writable(runtime_gate.parent, "运行时 lerobot 源码不可写，无法兜底打补丁。")
        _apply_train_gate_text_patch(runtime_gate)
        print(f"已把训练闸门豁免应用到运行时 lerobot（{runtime_src}）")
    except (PermissionError, RuntimeError) as exc:  # noqa: BLE001
        print(f"[警告] 运行时 lerobot 兜底打补丁失败：{exc}")


def ensure_model_device_gate_patch(model_dir: Path, *, allow: bool) -> None:
    """让便携 isaac_0_5 模型接受非 H100（如 A100/SM80）训练设备。

    模型远程代码 modeling_isaac05.py::require_qualified_runtime 强制
    transformers==5.5.4 / CUDA 12.8 / SM90(H100)。训练时冻结骨干只做 BF16 前向，
    与生产推理的 parity 要求无关，因此 opt-in 地跳过 capability/设备名检查：

        ISAAC_ALLOW_UNQUALIFIED_DEVICE=1 时 require_qualified_runtime 提前返回

    transformers 与 CUDA 版本检查保留（仍须用生产镜像）。补丁是幂等的文本替换，
    重复运行安全；模型目录不是 git 仓库，因此不做 git apply。
    """
    if not _is_portable_isaac05_repository(model_dir):
        print(f"[提示] {model_dir} 不是便携 isaac_0_5 仓库，跳过模型设备闸门 patch")
        return
    gate_file = model_dir / MODEL_GATE_FILE_NAME
    if not gate_file.is_file():
        raise FileNotFoundError(f"找不到模型远程代码: {gate_file}")
    text = gate_file.read_text(encoding="utf-8")
    if MODEL_GATE_MARKER in text:
        print(f"{gate_file} 已含 A100 训练豁免（{MODEL_GATE_ENV}=1 生效）")
        return
    if not allow:
        raise RuntimeError(
            f"A100(SM80) 等非 H100 训练需要设备闸门豁免：请加 --allow-unqualified-device "
            f"（脚本会向 {gate_file} 写入 {MODEL_GATE_ENV} 环境开关）。"
        )
    if MODEL_GATE_INSERT_ANCHOR not in text:
        raise RuntimeError(f"{gate_file} 的 require_qualified_runtime 结构与预期不符，请手工处理")
    require_writable(
        gate_file,
        "写入 A100 训练豁免需要模型挂载目录可写（modeling_isaac05.py）。",
    )
    text = text.replace(MODEL_GATE_INSERT_ANCHOR, MODEL_GATE_INSERT_REPLACEMENT, 1)
    if MODEL_GATE_IMPORT_ANCHOR in text:
        text = text.replace(MODEL_GATE_IMPORT_ANCHOR, MODEL_GATE_IMPORT_REPLACEMENT, 1)
    gate_file.write_text(text, encoding="utf-8")
    if MODEL_GATE_MARKER not in gate_file.read_text(encoding="utf-8"):
        raise RuntimeError("设备闸门 patch 应用后校验失败")
    print(f"已给 {gate_file} 写入 A100 训练豁免（{MODEL_GATE_ENV}=1 生效）")


# ---------------------------------------------------------------------------
# RTC（实时分块）训练支持：给模型侧 DiTActionExpertHead.forward 加 rtc_prefix_mask
# ---------------------------------------------------------------------------
RTC_FWD_FILE_NAME = "modeling_qwen35_vla.py"
RTC_FWD_MARKER = "# [PATCH] RTC training prefix"
RTC_FWD_SIG_ANCHOR = (
    "    def forward(\n"
    "        self,\n"
    "        vlm_activations: torch.Tensor,\n"
    "        vlm_mask: torch.Tensor | None,\n"
    "        x_tau: torch.Tensor,\n"
    "        tau: torch.Tensor,\n"
    "        horizon_emb: torch.Tensor | None = None,\n"
    "        action_mask: torch.Tensor | None = None,\n"
    "    ) -> torch.Tensor:\n"
)
RTC_FWD_SIG_REPLACEMENT = (
    "    def forward(\n"
    "        self,\n"
    "        vlm_activations: torch.Tensor,\n"
    "        vlm_mask: torch.Tensor | None,\n"
    "        x_tau: torch.Tensor,\n"
    "        tau: torch.Tensor,\n"
    "        horizon_emb: torch.Tensor | None = None,\n"
    "        action_mask: torch.Tensor | None = None,\n"
    f"        rtc_prefix_mask: torch.Tensor | None = None,  {RTC_FWD_MARKER}\n"
    "    ) -> torch.Tensor:\n"
)
RTC_FWD_BODY_ANCHOR = (
    "        if x_tau.dim() == 3:\n"
    "            return self.action_expert.forward_with_context(x_tau, self._flow_time(tau), context=context)\n"
    "        outs = [\n"
    "            self.action_expert.forward_with_context(x_tau[k], self._flow_time(tau[k]), context=context)\n"
    "            for k in range(sample_count)\n"
    "        ]\n"
    "        return torch.stack(outs, dim=0)\n"
)
RTC_FWD_BODY_REPLACEMENT = (
    "        # [PATCH] RTC training prefix: 前缀行用 clean 时间 1 的 AdaLN 条件，后缀行用采样时间\n"
    "        def _forward_with_rtc(xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:\n"
    "            rtc_conditioning = None\n"
    "            if rtc_prefix_mask is not None:\n"
    "                rtc_conditioning = self.action_expert.prepare_rtc_conditioning(t, rtc_prefix_mask)\n"
    "            return self.action_expert.forward_with_context(\n"
    "                xt, t, context=context, rtc_conditioning=rtc_conditioning\n"
    "            )\n"
    "        if x_tau.dim() == 3:\n"
    "            return _forward_with_rtc(x_tau, self._flow_time(tau))\n"
    "        outs = [_forward_with_rtc(x_tau[k], self._flow_time(tau[k])) for k in range(sample_count)]\n"
    "        return torch.stack(outs, dim=0)\n"
)


def ensure_rtc_training_patch(model_dir: Path, *, allow: bool = True) -> None:
    """让模型侧 DiTActionExpertHead.forward 支持 RTC 训练前缀（rtc_prefix_mask）。

    幂等文本补丁：加一个参数并在有前缀时构造 rtc_conditioning（前缀行=clean 时间 1，
    后缀行=采样时间），与 loss 侧 _flow_loss 的 RTC 采样配套。
    """
    if not _is_portable_isaac05_repository(model_dir):
        print(f"[提示] {model_dir} 不是便携 isaac_0_5 仓库，跳过 RTC forward patch")
        return
    fwd_file = model_dir / RTC_FWD_FILE_NAME
    if not fwd_file.is_file():
        raise FileNotFoundError(f"找不到模型建模文件: {fwd_file}")
    text = fwd_file.read_text(encoding="utf-8")
    if RTC_FWD_MARKER in text:
        print(f"{fwd_file} 已含 RTC forward 支持")
        return
    if not allow:
        raise RuntimeError("RTC 训练需要模型侧 forward 补丁：请允许脚本自动 patch modeling_qwen35_vla.py")
    if RTC_FWD_SIG_ANCHOR not in text or RTC_FWD_BODY_ANCHOR not in text:
        raise RuntimeError(
            f"{fwd_file} 的 DiTActionExpertHead.forward 结构与预期不符，请手工处理"
        )
    require_writable(fwd_file, "写入 RTC forward 支持需要模型目录可写。")
    text = text.replace(RTC_FWD_SIG_ANCHOR, RTC_FWD_SIG_REPLACEMENT, 1)
    text = text.replace(RTC_FWD_BODY_ANCHOR, RTC_FWD_BODY_REPLACEMENT, 1)
    fwd_file.write_text(text, encoding="utf-8")
    if RTC_FWD_MARKER not in fwd_file.read_text(encoding="utf-8"):
        raise RuntimeError("RTC forward patch 应用后校验失败")
    print(f"已给 {fwd_file} 写入 RTC 训练前缀支持")


def _is_portable_isaac05_repository(model_dir: Path) -> bool:
    """便携包布局：目录下 config.json 的 model_type == 'isaac_0_5'。"""
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return False
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return isinstance(raw, dict) and raw.get("model_type") == "isaac_0_5"


# ---------------------------------------------------------------------------
# 训练启动
# ---------------------------------------------------------------------------
def _build_wandb(args) -> tuple[dict[str, str] | None, dict[str, str]]:
    """根据 CLI 参数构造 (传给 lerobot_train 的 wandb 配置, 子进程 wandb 环境变量)。

    返回 (wandb_config, wandb_env)；--wandb 未开启时返回 (None, {})。
    """
    if not args.wandb:
        return None, {}
    mode = args.wandb_mode or ("online" if os.environ.get("WANDB_API_KEY") or args.wandb_api_key else "offline")
    config = {"project": args.wandb_project or "isaac05-finetune", "mode": mode}
    if args.wandb_entity:
        config["entity"] = args.wandb_entity
    if args.wandb_run_id:
        config["run_id"] = args.wandb_run_id
    env: dict[str, str] = {"WANDB_MODE": mode, "WANDB_SILENT": "True"}
    if args.wandb_api_key:
        env["WANDB_API_KEY"] = args.wandb_api_key
    if args.wandb_host:
        env["WANDB_BASE_URL"] = args.wandb_host
    if args.wandb_entity:
        env["WANDB_ENTITY"] = args.wandb_entity
    if args.wandb_run_id:
        env["WANDB_RUN_ID"] = args.wandb_run_id
    return config, env


def build_train_command(
    *,
    patch_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
    num_processes: int,
    batch_size: int,
    grad_accum: int,
    steps: int,
    save_freq: int,
    wandb: dict[str, str] | None = None,
    max_seq_len: int | None = None,
    rtc_max_delay_steps: int = 0,
    rtc_probability: float | None = None,
    rtc_delay_sampling: str = "uniform",
    rtc_poisson_mean: float = 5.0,
    rtc_prefix_length: int = 0,
    train_storage_fp32: bool = True,
    train_samples_per_chunk: int | None = None,
    fsdp: bool = False,
    fsdp_layer_cls: str = "Isaac05Qwen36DecoderLayer",
    log_freq: int = 200,
    num_workers: int | None = None,
    fsdp_backward_prefetch: str = "backward_pre",
    fsdp_forward_prefetch: bool = True,
    fsdp_activation_checkpointing: bool = False,
    torch_compile: bool = False,
    video_backend: str = "pyav",
    lr: float | None = None,
    action_expert_lr: float | None = None,
    vit_lr: float | None = None,
    max_train_steps: int | None = None,
    warmup_steps: int | None = None,
    loss_plan: str | None = None,
    detach_vlm_activations: bool = False,
) -> list[str]:
    """构造 accelerate + lerobot_train 命令。

    注意：传给 lerobot_train 的参数必须用 ``--key=value`` 等号形式 —— lerobot 的
    parser（configs/parser.py）对 --policy.path / --policy.type 等伪参数只按等号
    形式从 sys.argv 提取和过滤；空格分隔会被 draccus 判为 unrecognized arguments。
    """
    train_args: list[str] = [
        f"--policy.path={patch_dir}",
        "--dataset.repo_id=local/bimanual",
        f"--dataset.root={dataset_dir}",
        f"--dataset.video_backend={video_backend}",
        "--policy.train_expert_only=true",
        f"--batch_size={batch_size}",
        f"--gradient_accumulation_steps={grad_accum}",
        "--ddp_find_unused_parameters=false",
        f"--steps={steps}",
        "--save_checkpoint=true",
        f"--save_freq={save_freq}",
        f"--output_dir={output_dir}",
        f"--log_freq={int(log_freq)}",
    ]
    if num_workers is not None:
        train_args.append(f"--num_workers={int(num_workers)}")
    if max_seq_len is not None:
        train_args.append(f"--policy.train_max_sequence_length={int(max_seq_len)}")
    if rtc_max_delay_steps > 0:
        train_args.append(f"--policy.flow_rtc_max_delay_steps={int(rtc_max_delay_steps)}")
        train_args.append(f"--policy.flow_rtc_delay_sampling={str(rtc_delay_sampling)}")
        train_args.append(f"--policy.rtc_poisson_mean={float(rtc_poisson_mean)}")
        if rtc_probability is not None:
            train_args.append(f"--policy.rtc_probability={float(rtc_probability)}")
    if rtc_prefix_length > 0:
        train_args.append(f"--policy.rtc_prefix_length={int(rtc_prefix_length)}")
    if not train_storage_fp32 or fsdp:
        # FSDP 要求同一分片单元内 dtype 一致（可训练专家 FP32 + 冻结骨干 BF16 会报
        # "Must flatten tensors with uniform dtype"）。FSDP 下强制 train_storage_fp32=false
        # （全部 BF16），保存 FP32 用 --save-fp32 落盘转换。
        train_args.append("--policy.train_storage_fp32=false")
    if train_samples_per_chunk is not None:
        train_args.append(f"--policy.train_samples_per_chunk={int(train_samples_per_chunk)}")
    # ---- 学习率 / 调度覆盖 ----
    if lr is not None:
        train_args.append(f"--policy.optimizer_lr={float(lr)}")
    if action_expert_lr is not None:
        train_args.append(f"--policy.optimizer_action_expert_lr={float(action_expert_lr)}")
    if vit_lr is not None:
        train_args.append(f"--policy.optimizer_vit_lr={float(vit_lr)}")
    if max_train_steps is not None:
        train_args.append(f"--policy.max_train_steps={int(max_train_steps)}")
    if warmup_steps is not None:
        train_args.append(f"--policy.optimizer_warmup_steps={int(warmup_steps)}")
    if loss_plan is not None:
        train_args.append(f"--policy.loss_plan={str(loss_plan)}")
    if detach_vlm_activations:
        train_args.append("--policy.flow_matching_detach_vlm_activations=true")
    if wandb:
        train_args += [
            "--wandb.enable=true",
            f"--wandb.project={wandb['project']}",
        ]
        if wandb.get("entity"):
            train_args.append(f"--wandb.entity={wandb['entity']}")
        if wandb.get("run_id"):
            train_args.append(f"--wandb.run_id={wandb['run_id']}")
        if wandb.get("mode"):
            train_args.append(f"--wandb.mode={wandb['mode']}")
    # accelerate launch 显式传参：避免 "values were not passed ... defaults used" 警告
    # （accelerate 1.14 在 num_processes>1 且未显式 --multi_gpu 时也会提示多卡启用）
    if fsdp:
        # FSDP：模型分片（FULL_SHARD = 参数/梯度/优化器都分片），auto_wrap 按 transformer 层包裹。
        # use_orig_params=true：保留原始参数，让 perceptron_isaac 的冻结/FP32 提升在包装后可用。
        # state_dict_type=FULL_STATE_DICT：保存/加载整模型（checkpoint 可独立加载）。
        # backward_prefetch/forward_prefetch：隐藏通信延迟，提速。
        launch_args: list[str] = [
            f"--num_processes={num_processes}",
            "--num_machines=1",
            "--use_fsdp",
            "--fsdp_sharding_strategy",
            "FULL_SHARD",
            "--fsdp_auto_wrap_policy",
            "TRANSFORMER_BASED_WRAP",
            "--fsdp_transformer_layer_cls_to_wrap",
            str(fsdp_layer_cls),
            "--fsdp_state_dict_type",
            "FULL_STATE_DICT",
            "--fsdp_use_orig_params",
            "true",
            "--fsdp_backward_prefetch",
            str(fsdp_backward_prefetch).upper(),
            "--fsdp_forward_prefetch",
            "true" if fsdp_forward_prefetch else "false",
        ]
        if fsdp_activation_checkpointing:
            launch_args += ["--fsdp_activation_checkpointing", "true"]
        if torch_compile:
            launch_args += ["--dynamo_backend=inductor"]
        launch_args += ["--mixed_precision=bf16"]
    else:
        launch_args: list[str] = [
            f"--num_processes={num_processes}",
            "--num_machines=1",
            "--multi_gpu",
            "--mixed_precision=bf16",
            "--dynamo_backend=inductor" if torch_compile else "--dynamo_backend=no",
        ]
    return [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        *launch_args,
        "-m",
        "lerobot.scripts.lerobot_train",
        *train_args,
    ]


def launch_training(
    cmd: list[str],
    *,
    gpus: str,
    vocab: str | None,
    allow_unqualified_device: bool = False,
    wandb_env: dict[str, str] | None = None,
    log_path: Path | None = None,
    fsdp: bool = False,
) -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpus
    if vocab:
        env["QWEN35_VOCAB_PATH"] = vocab
    if allow_unqualified_device:
        env[MODEL_GATE_ENV] = "1"
    if wandb_env:
        env.update(wandb_env)
    if fsdp:
        # FSDP 分片训练：模型先加载到 CPU，由 FSDP 包装时按 rank 分片到各卡（避免整模型上 GPU OOM）
        env["ISAAC_LOAD_TO_CPU"] = "1"
    # 降低 CUDA 显存碎片（36B 在 80G 上非常紧张）
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # 让挂载仓库的 lerobot 优先于镜像快照（如 /lerobot），保证自动补丁实际生效
    mounted_src = str((REPO_ROOT / "lerobot" / "src").resolve())
    existing_pythonpath = env.get("PYTHONPATH", "")
    if mounted_src not in [entry for entry in existing_pythonpath.split(os.pathsep) if entry]:
        env["PYTHONPATH"] = os.pathsep.join(
            [mounted_src] + ([existing_pythonpath] if existing_pythonpath else [])
        )
        print(f"[PYTHONPATH] 前置挂载仓库 lerobot: {mounted_src}")
    print("=" * 78)
    print("训练命令:")
    print("  " + " ".join(cmd))
    print(
        f"  CUDA_VISIBLE_DEVICES={gpus} QWEN35_VOCAB_PATH={env.get('QWEN35_VOCAB_PATH')} "
        f"{MODEL_GATE_ENV}={env.get(MODEL_GATE_ENV, '')}"
    )
    if wandb_env:
        print(f"  wandb: {wandb_env}")
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  训练日志(离线可视化): {log_path}")
    print("=" * 78)
    if log_path is None:
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            sys.exit(f"训练失败，退出码 {result.returncode}")
    else:
        # tee：stdout+stderr 同时输出到终端和日志文件，供 8600 离线可视化解析
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        with open(log_path, "a", encoding="utf-8") as logf:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                logf.write(line)
        returncode = proc.wait()
        if returncode != 0:
            sys.exit(f"训练失败，退出码 {returncode}")
    print("训练完成。")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Isaac 0.5 双臂 EEF 流匹配微调（LeRobot v3.0 数据集）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir", default=None,
                        help="数据集根目录（含 meta/ data/ videos/）；--resume 续训时可省略")
    parser.add_argument("--base-package", default=None,
                        help="导入的 Isaac-0.5 LeRobot 包（只读源）；默认自动探测容器挂载点")
    parser.add_argument("--patch-dir", default=None,
                        help="可写 patch/训练副本；默认放在模型目录同级 lerobot_policy_patched")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "isaac-finetune"),
                        help="检查点输出目录（默认 <仓库>/outputs/isaac-finetune）")
    parser.add_argument("--vocab", default=None,
                        help="Qwen35Harmony 词表 vocab.json（默认取 QWEN35_VOCAB_PATH，再探测 /vocab 等挂载点）")

    parser.add_argument("--action-dim", type=int, default=20, help="动作维度（双臂 EEF）")
    parser.add_argument("--proprio-dim", type=int, default=34, help="状态维度")
    parser.add_argument("--cameras", default="head,left,right", help="相机键（与 meta features 一致）")
    parser.add_argument("--image-size", default="240,424", help="H,W（与 meta 图像 shape 一致）")
    parser.add_argument("--fps", type=float, default=10.0, help="数据集/模型时钟 fps")
    parser.add_argument("--n-obs-steps", type=int, default=None, choices=[1, 3],
                        help="观测历史帧数（默认 3；3 相机+大分辨率导致序列超 4096 时可改 1，省 2/3 图像 token）")
    parser.add_argument("--max-seq-len", type=int, default=None,
                        help="覆盖 policy.train_max_sequence_length（默认 4096；序列超长时提高，如 8192，注意显存）")
    parser.add_argument("--train-storage-fp32", type=lambda s: s.lower() in ("1", "true", "yes"),
                        default=True,
                        help="可训练参数 FP32 存储（A100-80G 装 36B 权重紧张时设 false 用 BF16，省 ~4GB，冒烟可用）")
    parser.add_argument("--train-samples-per-chunk", type=int, default=None,
                        help="每 chunk 的 flow MC 采样数（默认 8；显存紧张时改 1，省激活显存）")
    parser.add_argument("--save-fp32", action="store_true",
                        help="训练结束后把 checkpoints/last/pretrained_model 的模型权重转成 FP32 保存"
                             "（配合 --train-storage-fp32=false 使用：训练省显存、落盘保 FP32）")
    parser.add_argument("--log-freq", type=int, default=None,
                        help="每多少步打印一次 loss 并同步 wandb（默认：冒烟 5、正式 200；"
                             "注意 lerobot 默认 200，冒烟 20 步会一条都不打）")
    # ---- 训练提速 ----
    parser.add_argument("--num-workers", type=int, default=None,
                        help="dataloader 进程数（默认 lerobot=4；视频解码慢时调 8-16 提速）")
    parser.add_argument("--fsdp-backward-prefetch", default="backward_pre",
                        choices=["no_prefetch", "backward_pre", "backward_post"],
                        help="FSDP 反向预取策略（backward_pre 隐藏通信延迟，提速）")
    parser.add_argument("--fsdp-forward-prefetch", type=lambda s: s.lower() in ("1", "true", "yes"),
                        default=True,
                        help="FSDP 前向预取下一层参数（隐藏 all-gather 延迟，提速）")
    parser.add_argument("--fsdp-activation-checkpointing", action="store_true",
                        help="FSDP 激活检查点（省显存；显存宽裕时不开更快）")
    parser.add_argument("--torch-compile", action="store_true",
                        help="用 torch.compile(inductor) 加速（潜在 1.3-1.5×，但 remote-code+FSDP 有兼容风险，"
                             "建议单独小实验验证；= 加 --dynamo_backend=inductor）")
    parser.add_argument("--video-backend", default="pyav", choices=["pyav", "torchcodec"],
                        help="视频解码后端（torchcodec 更快，但生产 torch2.10 不匹配；当前容器 torch2.11 可试）")
    # ---- 学习率 / 调度 ----
    parser.add_argument("--lr", type=float, default=None,
                        help="覆盖 policy.optimizer_lr（VLM 骨干 LR；train_expert_only 下不生效）")
    parser.add_argument("--action-expert-lr", type=float, default=None,
                        help="覆盖 policy.optimizer_action_expert_lr（动作头 LR，train_expert_only 实际生效的；"
                             "默认 5e-5；有效 batch 大（128）可提到 1e-4）")
    parser.add_argument("--vit-lr", type=float, default=None,
                        help="覆盖 policy.optimizer_vit_lr（视觉 LR；train_expert_only 下不生效）")
    parser.add_argument("--max-train-steps", type=int, default=None,
                        help="覆盖 policy.max_train_steps（cosine 衰减总步数；建议 = 训练 steps，默认 30000 会导致"
                             "20000 步训练时 LR 没衰减完）")
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="覆盖 policy.optimizer_warmup_steps（默认 200）")
    parser.add_argument("--loss-plan", default=None,
                        choices=["text_ntp_fast_flow_action", "flow_matching_action_prediction"],
                        help="loss_plan：默认 text_ntp_fast_flow_action（文本NTP+流匹配）；"
                             "train_expert_only 下文本目标不更新任何参数、纯浪费，建议 "
                             "flow_matching_action_prediction（纯流匹配，总 loss 即 flow_loss）")
    parser.add_argument("--detach-vlm-activations", action="store_true",
                        help="冻结骨干时把 VLM 激活 detach（强烈建议）：反向只在动作头内计算，"
                             "不再回传 36B 骨干 → 步时大降、显存大降；数学上与不 detach 等价"
                             "（骨干无可训练参数，激活梯度本就会被丢弃）")
    parser.add_argument("--fsdp", action="store_true",
                        help="用 FSDP 分片训练（等价 ZeRO-3，解决 36B 单卡放不下的问题；"
                             "自动 --fsdp full_shard auto_wrap + 模型加载到 CPU 由 FSDP 分片）")
    parser.add_argument("--fsdp-layer-cls", default="Isaac05Qwen36DecoderLayer",
                        help="FSDP auto_wrap 的 transformer 层类名")
    # ---- RTC（实时分块）训练/推理 ----
    parser.add_argument("--rtc-max-delay-steps", type=int, default=0,
                        help="RTC 最大前缀步数（基础模型为 12；>0 且 --rtc-probability>0 开启 RTC 训练）")
    parser.add_argument("--rtc-probability", type=float, default=None,
                        help="每个 chunk 应用 RTC 前缀的概率（基础模型 0.5）")
    parser.add_argument("--rtc-delay-sampling", default="uniform",
                        choices=["uniform", "exponential", "poisson"])
    parser.add_argument("--rtc-poisson-mean", type=float, default=5.0)
    parser.add_argument("--rtc-prefix-length", type=int, default=0,
                        help="推理时每次携带的已执行动作前缀行数（需 rtc_probability>0 且 <= chunk_size-1）")

    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="训练 GPU（CUDA_VISIBLE_DEVICES）")
    parser.add_argument("--num-processes", type=int, default=None, help="accelerate 进程数（默认=GPU 数）")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--steps", type=int, default=None,
                        help="训练总步数（默认 20000；--resume 时可选覆盖续训目标步数）")
    parser.add_argument("--save-freq", type=int, default=2500, help="检查点保存间隔")
    parser.add_argument("--resume", action="store_true",
                        help="从检查点续训：--resume-config 指向 train_config.json（默认 <output>/checkpoints/last/train_config.json）")
    parser.add_argument("--resume-config", default=None, help="续训用 train_config.json 路径")
    parser.add_argument("--smoke", action="store_true", help="冒烟模式：20 步、grad-accum 8、输出到 <output>/smoke")
    parser.add_argument("--overwrite", action="store_true",
                        help="非 resume 时若输出目录已存在，先删除再训练（注意：会清空该目录，仅用于重跑冒烟等场景）")
    parser.add_argument("--skip-patch-src", action="store_true",
                        help="不自动应用 lerobot expert-only 训练补丁（源码已打补丁时用）")
    parser.add_argument("--allow-unqualified-device", action="store_true",
                        help="A100(SM80) 等非 H100 训练：向便携模型写入设备闸门豁免 "
                             "（env ISAAC_ALLOW_UNQUALIFIED_DEVICE=1，transformers/CUDA 版本检查保留）")

    # ---- wandb 可视化 ----
    parser.add_argument("--wandb", action="store_true", help="启用 wandb 训练可视化")
    parser.add_argument("--wandb-project", default="isaac05-finetune", help="wandb 项目名")
    parser.add_argument("--wandb-entity", default=None, help="wandb 实体（团队/用户名）")
    parser.add_argument("--wandb-run-id", default=None, help="wandb run id（续记同一 run）")
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"],
                        help="wandb 模式：默认 WANDB_API_KEY 已设->online，否则 offline")
    parser.add_argument("--wandb-api-key", default=None, help="wandb API key（也设 WANDB_API_KEY）")
    parser.add_argument("--wandb-host", default=None,
                        help="wandb 服务地址（如本地面板 http://localhost:8600，设 WANDB_BASE_URL）")
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    base_package = Path(args.base_package).resolve() if args.base_package else detect_base_package()
    print(f"基础包: {base_package}")

    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    image_size = [int(x) for x in args.image_size.split(",")]
    if len(image_size) != 2:
        raise ValueError("--image-size 需要 H,W 两个值")
    gpus = [g for g in args.gpus.split(",") if g.strip()]
    num_processes = args.num_processes or len(gpus)
    output_dir = Path(args.output_dir).resolve()
    if args.smoke:
        steps, grad_accum = 20, 8
        output_dir = output_dir / "smoke"
        print("[smoke] 冒烟模式：20 步短跑验证")
    else:
        steps, grad_accum = (args.steps or 20000), args.grad_accum

    # LR 衰减长度默认跟随训练步数（--steps 一个旋钮即可；--max-train-steps 可显式覆盖）
    max_train_steps = args.max_train_steps if args.max_train_steps is not None else steps

    # lerobot trainer 非 resume 时拒绝已存在的输出目录；显式 --overwrite 才清空重跑
    if args.overwrite and not args.resume and output_dir.is_dir():
        print(f"[overwrite] 删除已存在的输出目录: {output_dir}")
        shutil.rmtree(output_dir)
    if not args.resume and not args.overwrite and output_dir.is_dir():
        raise SystemExit(
            f"输出目录已存在: {output_dir}\n"
            "lerobot 训练器非 resume 模式拒绝覆盖（会直接 FileExistsError）。二选一：\n"
            f"  1) 删除旧目录后重跑: rm -rf {output_dir}\n"
            "  2) 重跑时加 --overwrite（脚本会先清空该目录）\n"
            "注意：正式训练中断后请用 --resume 续训，不要用 --overwrite。"
        )

    vocab = args.vocab or os.environ.get("QWEN35_VOCAB_PATH") or detect_vocab()
    if not vocab:
        print("[提示] 未设置 QWEN35_VOCAB_PATH 且未探测到词表，mharmony 词表加载可能失败")
    else:
        print(f"词表: {vocab}")

    model_parent = base_package.parent

    # ---- [0] 续训分支：直接复用检查点的 train_config.json ----
    if args.resume:
        print("== [resume] 从检查点续训 ==")
        resume_cfg = Path(args.resume_config or (output_dir / "checkpoints" / "last" / "train_config.json"))
        if not resume_cfg.is_file():
            raise FileNotFoundError(f"续训配置不存在: {resume_cfg}")
        ensure_expert_only_training_patch(apply=not args.skip_patch_src)
        allow_unqualified = args.allow_unqualified_device or os.environ.get(MODEL_GATE_ENV) == "1"
        if allow_unqualified:
            ensure_model_device_gate_patch(model_parent, allow=True)
        if args.fsdp:
            launch_prefix = [
                sys.executable, "-m", "accelerate.commands.launch",
                f"--num_processes={num_processes}",
                "--num_machines=1",
                "--use_fsdp",
                "--fsdp_sharding_strategy", "FULL_SHARD",
                "--fsdp_auto_wrap_policy", "TRANSFORMER_BASED_WRAP",
                "--fsdp_transformer_layer_cls_to_wrap", args.fsdp_layer_cls,
                "--fsdp_state_dict_type", "FULL_STATE_DICT",
                "--fsdp_use_orig_params", "true",
                "--fsdp_backward_prefetch", str(args.fsdp_backward_prefetch).upper(),
                "--fsdp_forward_prefetch", "true" if args.fsdp_forward_prefetch else "false",
                "--mixed_precision=bf16",
                "-m", "lerobot.scripts.lerobot_train",
            ]
            if args.fsdp_activation_checkpointing:
                launch_prefix.insert(-2, "--fsdp_activation_checkpointing")
                launch_prefix.insert(-2, "true")
            if args.torch_compile:
                launch_prefix.insert(-2, "--dynamo_backend=inductor")
        else:
            launch_prefix = [
                sys.executable, "-m", "accelerate.commands.launch",
                f"--num_processes={num_processes}",
                "--num_machines=1",
                "--multi_gpu",
                "--mixed_precision=bf16",
                "--dynamo_backend=inductor" if args.torch_compile else "--dynamo_backend=no",
                "-m", "lerobot.scripts.lerobot_train",
            ]
        cmd = [
            *launch_prefix,
            f"--config_path={resume_cfg}",
            "--resume=true",
            f"--dataset.video_backend={args.video_backend}",
        ]
        if args.num_workers is not None:
            cmd.append(f"--num_workers={int(args.num_workers)}")
        if args.fsdp:
            # FSDP 要求同一分片单元 dtype 一致：续训同样强制 BF16 存储（保存 FP32 用 --save-fp32）
            cmd.append("--policy.train_storage_fp32=false")
        if args.steps is not None:
            cmd.append(f"--steps={args.steps}")
            # LR 衰减长度跟随续训目标步数
            cmd.append(f"--policy.max_train_steps={args.steps}")
        elif args.max_train_steps is not None:
            cmd.append(f"--policy.max_train_steps={args.max_train_steps}")
        wandb_config, wandb_env = _build_wandb(args)
        if wandb_config:
            cmd += [
                "--wandb.enable=true",
                f"--wandb.project={wandb_config['project']}",
                f"--wandb.mode={wandb_config['mode']}",
            ]
            if wandb_config.get("entity"):
                cmd.append(f"--wandb.entity={wandb_config['entity']}")
            if wandb_config.get("run_id"):
                cmd.append(f"--wandb.run_id={wandb_config['run_id']}")
        launch_training(
            cmd, gpus=",".join(gpus), vocab=vocab,
            allow_unqualified_device=allow_unqualified,
            wandb_env=wandb_env or None,
            log_path=output_dir.parent / f"{output_dir.name}.train.log",
            fsdp=args.fsdp,
        )
        if args.save_fp32:
            _convert_weights_to_fp32(output_dir / "checkpoints" / "last" / "pretrained_model")
        print(f"续训完成，检查点: {output_dir / 'checkpoints' / 'last' / 'pretrained_model'}")
        return

    if not args.dataset_dir:
        raise SystemExit("--dataset-dir 必填（非 --resume 模式）")
    dataset_dir = Path(args.dataset_dir).resolve()

    # ---- 1. 数据集校验 ----
    print("== [1/4] 数据集校验 ==")
    info = load_dataset_info(dataset_dir)
    validate_dataset(
        dataset_dir, info,
        action_dim=args.action_dim, proprio_dim=args.proprio_dim,
        cameras=cameras, fps=args.fps,
    )

    # ---- 2. 可写副本 + 几何 patch + 统计量 ----
    print("== [2/4] patch 可写副本 ==")
    patch_dir = Path(args.patch_dir).resolve() if args.patch_dir else (model_parent / "lerobot_policy_patched")
    if patch_dir.parent != model_parent:
        print(
            f"[警告] patch 目录 {patch_dir} 与模型目录 {model_parent} 不同级；"
            "config.json 的 hf_model_path='..' 将无法解析为便携 isaac_0_5 仓库，"
            "from_pretrained 会报 'escapes the package root'。请将 --patch-dir 放在模型目录同级。"
        )
    patch_dir = prepare_package(base_package, patch_dir)

    # 动作目标长度沿用基础包的 chunk_size（MK1 契约要求 chunk_size <= action_horizon=50）
    chunk_size = int(json.loads((patch_dir / "config.json").read_text(encoding="utf-8")).get("chunk_size", 50))
    warn_short_episodes(dataset_dir, chunk_size)

    info_features = info.get("features") or {}
    action_names = _flatten_names(info_features.get("action", {}).get("names")) or [
        f"a_{i}" for i in range(args.action_dim)
    ]
    state_names = _flatten_names(info_features.get("observation.state", {}).get("names")) or [
        f"s_{i}" for i in range(args.proprio_dim)
    ]
    if len(action_names) != args.action_dim or len(state_names) != args.proprio_dim:
        raise ValueError(
            f"meta 特征名数量不匹配: action={len(action_names)}/{args.action_dim}, "
            f"state={len(state_names)}/{args.proprio_dim}"
        )
    patch_package(
        patch_dir,
        action_dim=args.action_dim, proprio_dim=args.proprio_dim,
        cameras=cameras, image_size=image_size, fps=args.fps,
        action_names=action_names, state_names=state_names,
        n_obs_steps=args.n_obs_steps,
        rtc_max_delay_steps=args.rtc_max_delay_steps,
        rtc_probability=args.rtc_probability,
        rtc_delay_sampling=args.rtc_delay_sampling,
        rtc_poisson_mean=args.rtc_poisson_mean,
        rtc_prefix_length=args.rtc_prefix_length,
    )

    arrays = load_vector_columns(dataset_dir, list(STATS_FEATURES))
    if arrays["action"].shape[1] != args.action_dim:
        raise ValueError(f"action 数据宽 {arrays['action'].shape[1]} != {args.action_dim}")
    if arrays["observation.state"].shape[1] != args.proprio_dim:
        raise ValueError(f"state 数据宽 {arrays['observation.state'].shape[1]} != {args.proprio_dim}")
    write_isaac_stats(
        patch_dir,
        action_arr=arrays["action"], state_arr=arrays["observation.state"],
        chunk_size=chunk_size, fps=args.fps,
        action_dim=args.action_dim, proprio_dim=args.proprio_dim,
    )
    ensure_dataset_stats_json(dataset_dir, arrays)

    # ---- 3. lerobot 源码补丁 + （可选）A100 设备闸门豁免 ----
    print("== [3/4] 检查 lerobot expert-only 训练补丁 ==")
    ensure_expert_only_training_patch(apply=not args.skip_patch_src)

    allow_unqualified = args.allow_unqualified_device or os.environ.get(MODEL_GATE_ENV) == "1"
    if allow_unqualified:
        print(f"== [3b] A100/非 H100 训练：应用 {MODEL_GATE_ENV}=1 设备豁免 ==")
        ensure_model_device_gate_patch(model_parent, allow=True)

    rtc_enabled = args.rtc_max_delay_steps > 0 and (args.rtc_probability or 0) > 0
    if rtc_enabled:
        print(
            f"== [3c] RTC 训练：max_delay={args.rtc_max_delay_steps} "
            f"probability={args.rtc_probability} sampling={args.rtc_delay_sampling} =="
        )
        ensure_rtc_training_patch(model_parent, allow=True)

    # ---- 4. 训练 ----
    print("== [4/4] accelerate 多卡训练 ==")
    log_freq = args.log_freq if args.log_freq is not None else (5 if args.smoke else 200)
    wandb_config, wandb_env = _build_wandb(args)
    if wandb_config:
        print(f"== [4b] wandb 可视化已启用: project={wandb_config['project']} mode={wandb_config['mode']} ==")
    cmd = build_train_command(
        patch_dir=patch_dir, dataset_dir=dataset_dir, output_dir=output_dir,
        num_processes=num_processes, batch_size=args.batch_size,
        grad_accum=grad_accum, steps=steps, save_freq=args.save_freq,
        wandb=wandb_config,
        max_seq_len=args.max_seq_len,
        rtc_max_delay_steps=args.rtc_max_delay_steps,
        rtc_probability=args.rtc_probability,
        rtc_delay_sampling=args.rtc_delay_sampling,
        rtc_poisson_mean=args.rtc_poisson_mean,
        rtc_prefix_length=args.rtc_prefix_length,
        train_storage_fp32=args.train_storage_fp32,
        train_samples_per_chunk=args.train_samples_per_chunk,
        fsdp=args.fsdp,
        fsdp_layer_cls=args.fsdp_layer_cls,
        log_freq=log_freq,
        num_workers=args.num_workers,
        fsdp_backward_prefetch=args.fsdp_backward_prefetch,
        fsdp_forward_prefetch=args.fsdp_forward_prefetch,
        fsdp_activation_checkpointing=args.fsdp_activation_checkpointing,
        torch_compile=args.torch_compile,
        video_backend=args.video_backend,
        lr=args.lr,
        action_expert_lr=args.action_expert_lr,
        vit_lr=args.vit_lr,
        max_train_steps=max_train_steps,
        warmup_steps=args.warmup_steps,
        loss_plan=args.loss_plan,
        detach_vlm_activations=args.detach_vlm_activations,
    )
    launch_training(
        cmd, gpus=",".join(gpus), vocab=vocab,
        allow_unqualified_device=allow_unqualified,
        wandb_env=wandb_env or None,
        log_path=output_dir.parent / f"{output_dir.name}.train.log",
        fsdp=args.fsdp,
    )
    if args.save_fp32:
        _convert_weights_to_fp32(output_dir / "checkpoints" / "last" / "pretrained_model")

    print("=" * 78)
    print(f"检查点: {output_dir / 'checkpoints' / 'last' / 'pretrained_model'}")
    print("离线评测: python /algorithm/repo/docker/eval_offline.py \\")
    print(f"    --policy-path {output_dir / 'checkpoints' / 'last' / 'pretrained_model'} \\")
    print(f"    --dataset-repo-id local/bimanual --dataset-root {dataset_dir}")
    print("在线部署: python /algorithm/repo/run/deploy_isaac.py \\")
    print(f"    --policy-path {output_dir / 'checkpoints' / 'last' / 'pretrained_model'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
