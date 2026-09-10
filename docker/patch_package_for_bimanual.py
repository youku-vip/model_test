#!/usr/bin/env python
"""Patch an imported Isaac-0.5 LeRobot package for a custom bimanual EEF robot.

The action expert head is max_action_dim=64 wide and the proprio encoder pads
state up to vector_max_states=128, so a 20D action / 34D state robot trains by
re-training the action head (train_expert_only) on the custom dataset. This
script:

  1. patches <package>/config.json  (action_dim, proprio_dim, cameras, image, fps)
  2. recomputes <package>/isaac_stats.json  (q01/q99 per dim from the dataset)
  3. validates the dataset contract (features, camera keys, fps, dims)

Usage:
    python patch_package_for_bimanual.py \
        --package /data/isaac/model/lerobot_policy \
        --dataset-dir /path/to/dataset \
        --action-dim 20 --proprio-dim 34 \
        --cameras head,left,right \
        --image-size 240,424 \
        --fps 10
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_parquet_columns(dataset_dir: Path, cols=("action", "observation.state")):
    """Yield concatenated numpy arrays for each requested column."""
    data_files = sorted((dataset_dir / "data").glob("chunk-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"no parquet files under {dataset_dir / 'data'}")
    acc = {c: [] for c in cols}
    for pq in data_files:
        df = pd.read_parquet(pq, columns=list(cols))
        for c in cols:
            acc[c].append(np.asarray(df[c].tolist(), dtype=np.float32))
    return {c: np.concatenate(v) for c, v in acc.items()}


def quantile_stats(values: np.ndarray) -> tuple[list, list]:
    """Per-dimension q01/q99 (np.nanpercentile skips NaN frames)."""
    q01 = np.nanpercentile(values, 1.0, axis=0)
    q99 = np.nanpercentile(values, 99.0, axis=0)
    return q01.tolist(), q99.tolist()


def write_isaac_stats(package: Path, action_arr, state_arr, chunk_size, fps, action_dim, proprio_dim):
    a_q01, a_q99 = quantile_stats(action_arr)
    s_q01, s_q99 = quantile_stats(state_arr)
    stats = {
        "action": {"q01": a_q01, "q99": a_q99},
        "action_dim": int(action_dim),
        "action_horizon": int(chunk_size),
        "action_normalization_eps": 1e-06,
        "action_representation": "absolute",
        "clip_normalized_actions": True,
        "clip_normalized_max": 10.0,
        "profile_id": None,
        "profile_scope": None,
        "proprio": {"q01": s_q01, "q99": s_q99},
        "proprio_dim": int(proprio_dim),
        "proprio_normalization_eps": 1e-06,
        "relative_exclude_joints": [],
        "schema": "flow_matching_stats_v1",
        "state_action_schema": f"custom_bimanual_eef_{action_dim}",
        "stats_sha256": None,  # filled below
        "target_fps": float(fps),
        "validation_status": None,
    }
    digest = hashlib.sha256(
        json.dumps({k: v for k, v in stats.items() if k != "stats_sha256"},
                   sort_keys=True).encode()
    ).hexdigest()
    stats["stats_sha256"] = digest
    out = package / "isaac_stats.json"
    out.write_text(json.dumps(stats, indent=2))
    print(f"wrote {out} (action={action_dim}D, proprio={proprio_dim}D)")


def patch_config(package: Path, action_dim, proprio_dim, cameras, image_size, fps):
    cfg_path = package / "config.json"
    cfg = json.loads(cfg_path.read_text())
    action_feature_names = [f"a_{i}" for i in range(action_dim)]
    state_feature_names = [f"s_{i}" for i in range(proprio_dim)]

    cfg.update({
        "action_dim": int(action_dim),
        "proprio_dim": int(proprio_dim),
        "image_size": [int(x) for x in image_size],
        "camera_order": list(cameras),
        "target_fps": float(fps),
        "action_feature_names": action_feature_names,
        "state_feature_names": state_feature_names,
        # non-MK1 (isaac_0_5) load path does not hard-validate robot_type;
        # keep it descriptive for the custom robot.
        "robot_type": "custom_bimanual",
        "control_mode": "ee",
        # [PATCH] 本地训练不推 Hub：基础包 push_to_hub=true 且无 repo_id，train 校验会 raise。
        "push_to_hub": False,
        # [PATCH] 自定义 robot_type 不在 bi_yam/so100_so101/libero 白名单内，
        # 必须关闭严格部署契约校验（PerceptronIsaacConfig.__post_init__ 会 raise）。
        "strict_hardware_feature_contract": False,
        "strict_environment_feature_contract": False,
        # [PATCH] MK1 contract identity: portable isaac_0_5 flavor synthesizes
        # trained_steps=1; keep the package config in agreement.
        "trained_steps": 1,
    })
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"patched {cfg_path}: action_dim={action_dim}, proprio_dim={proprio_dim}, "
          f"cameras={cameras}, image_size={image_size}, fps={fps}")


def validate_dataset(dataset_dir: Path, action_dim, proprio_dim, cameras, fps):
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    feats = info["features"]
    state_shape = feats["observation.state"]["shape"][0]
    action_shape = feats["action"]["shape"][0]
    assert info["codebase_version"] == "v3.0", "dataset must be LeRobot v3.0"
    assert state_shape == proprio_dim, f"state dim {state_shape} != {proprio_dim}"
    assert action_shape == action_dim, f"action dim {action_shape} != {action_dim}"
    missing = [c for c in cameras if f"observation.images.{c}" not in feats]
    assert not missing, f"missing camera features: {missing}"
    assert info["fps"] == fps, f"dataset fps {info['fps']} != {fps}"
    print(f"dataset OK: v3.0, state={state_shape}D, action={action_shape}D, "
          f"cameras={cameras}, fps={info['fps']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True, help="imported package dir (lerobot_policy)")
    ap.add_argument("--dataset-dir", required=True, help="LeRobot v3 dataset dir")
    ap.add_argument("--action-dim", type=int, default=20)
    ap.add_argument("--proprio-dim", type=int, default=34)
    ap.add_argument("--cameras", default="head,left,right")
    ap.add_argument("--image-size", default="240,424")
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args()

    package = Path(args.package)
    dataset = Path(args.dataset_dir)
    cameras = [c.strip() for c in args.cameras.split(",")]
    image_size = [int(x) for x in args.image_size.split(",")]

    validate_dataset(dataset, args.action_dim, args.proprio_dim, cameras, args.fps)
    patch_config(package, args.action_dim, args.proprio_dim, cameras, image_size, args.fps)

    data = load_parquet_columns(dataset)
    cfg = json.loads((package / "config.json").read_text())
    write_isaac_stats(package, data["action"], data["observation.state"],
                      cfg.get("chunk_size", 50), args.fps,
                      args.action_dim, args.proprio_dim)
    print("\nDONE - 现在可以训练：")
    print("  lerobot-train --policy.path=<package> --dataset.repo_id=... "
          "--dataset.video_backend=pyav --policy.train_expert_only=true ...")


if __name__ == "__main__":
    main()
