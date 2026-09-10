#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeepSpeed ZeRO-3 launcher for the existing Isaac-0.5 finetune entrypoint.

This intentionally wraps ``run/finetune_isaac05.py`` instead of duplicating its
large dataset/patching pipeline. The base script still owns validation,
processor/model patching, checkpointing and LeRobot arguments; this wrapper
only rewrites the final Accelerate launch backend to DeepSpeed ZeRO-3.

Why ZeRO-3, not ZeRO-2?
-----------------------
Isaac-0.5 is a 36B-class model. In BF16, parameters alone are ~72 GB, before
runtime buffers and activations. ZeRO-2 partitions optimizer/gradient state but
keeps model parameters replicated, which is too tight for an A100-80GB. ZeRO-3
partitions parameters as well.

This is an opt-in benchmark backend. FSDP remains the default path in the base
script because ZeRO-3 checkpoint/save behavior and communication overhead need
to be measured on the exact pinned Perceptron LeRobot stack.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from run import finetune_isaac05 as base


DEFAULT_CONFIG = Path(__file__).resolve().with_name("deepspeed_zero3_a100.json")


def _rewrite_launch_command(cmd: list[str], config: Path) -> list[str]:
    """Replace Accelerate's FSDP/DDP launch backend with DeepSpeed ZeRO-3."""
    try:
        launch_idx = cmd.index("-m")
        if cmd[launch_idx + 1] != "accelerate.commands.launch":
            raise ValueError
    except (ValueError, IndexError):
        raise RuntimeError("finetune_isaac05.py 生成的 accelerate launch 命令结构发生变化")

    # Keep the target module + all LeRobot training args after the second -m.
    target_idx = None
    for i in range(launch_idx + 2, len(cmd) - 1):
        if cmd[i] == "-m" and cmd[i + 1] == "lerobot.scripts.lerobot_train":
            target_idx = i
            break
    if target_idx is None:
        raise RuntimeError("找不到 lerobot.scripts.lerobot_train，无法改写 DeepSpeed 启动命令")

    accelerator_args = [
        arg for arg in cmd[launch_idx + 2 : target_idx]
        if not arg.startswith("--mixed_precision=")
        and not arg.startswith("--multi_gpu")
        and not arg.startswith("--use_fsdp")
        and not arg.startswith("--fsdp_")
        and not arg.startswith("--dynamo_backend=")
    ]
    # With a DeepSpeed config file Accelerate expects precision/ZeRO settings
    # to live in that file rather than being duplicated on the CLI.
    accelerator_args += [
        "--use_deepspeed",
        f"--deepspeed_config_file={config}",
    ]
    return cmd[: launch_idx + 2] + accelerator_args + cmd[target_idx:]


def _patch_launch_training() -> None:
    original = base.launch_training

    def launch_training(cmd: list[str], **kwargs) -> None:  # type: ignore[no-untyped-def]
        config = Path(os.environ.get("ISAAC_DEEPSPEED_CONFIG", str(DEFAULT_CONFIG))).resolve()
        if not config.is_file():
            raise FileNotFoundError(f"DeepSpeed config 不存在: {config}")

        rewritten = _rewrite_launch_command(cmd, config)

        # ZeRO-3 constructs/shards massive parameters during distributed init.
        # Keep the same CPU-first safety posture as the base FSDP path.
        os.environ["ISAAC_LOAD_TO_CPU"] = "1"
        kwargs["fsdp"] = False
        print(f"[DeepSpeed] ZeRO-3 config: {config}")
        print("[DeepSpeed] 36B BF16 参数分片；未启用 CPU/NVMe offload。")
        return original(rewritten, **kwargs)

    base.launch_training = launch_training


def main() -> None:
    _patch_launch_training()
    # Preserve the original CLI exactly, so every existing argument remains
    # available (dataset checks, RTC, Flow MC, action LR, save-fp32, etc.).
    sys.argv[0] = str(Path(__file__).resolve())
    base.main()


if __name__ == "__main__":
    main()
