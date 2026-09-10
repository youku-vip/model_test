#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified Isaac-0.5 quality-first launcher.

Supports train scopes:
  expert                 action expert only
  expert_connector       action expert + vector/proprio connector
  expert_connector_lora  action expert + connector + VLM/vision LoRA

Supports backends:
  fsdp1        existing stable path
  fsdp2        Accelerate FSDP2 config
  deepspeed3   DeepSpeed ZeRO-3 config

This wrapper keeps the existing finetune_isaac05.py data/model pipeline and
applies small idempotent source edits to the pinned Perceptron LeRobot tree
before the distributed child processes are launched.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from run import finetune_isaac05 as base  # noqa: E402


QWEN35_FILE = REPO_ROOT / "lerobot/src/lerobot/policies/perceptron_isaac/modeling_qwen35_vla.py"
ISAAC_POLICY_FILE = REPO_ROOT / "lerobot/src/lerobot/policies/perceptron_isaac/modeling_perceptron_isaac.py"
FSDP2_CONFIG = Path(__file__).resolve().with_name("fsdp2_a100.yaml")
DEEPSPEED_CONFIG = Path(__file__).resolve().with_name("deepspeed_zero3_a100.json")


def _patch_text_once(path: Path, marker: str, replacements: list[tuple[str, str]]) -> None:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    original = text
    for old, new in replacements:
        if old not in text:
            raise RuntimeError(f"advanced patch anchor missing in {path}: {old[:120]!r}")
        text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")
    if marker not in text or text == original:
        raise RuntimeError(f"advanced patch verification failed for {path}")


def apply_train_scope_patch() -> None:
    # Qwen helper: upstream expert-only setup freezes everything but action_expert.
    # Re-enable the vector/proprio connector for the two connector scopes.
    _patch_text_once(
        QWEN35_FILE,
        "[PATCH] ISAAC advanced train scopes",
        [
            (
                "from pathlib import Path\n",
                "from pathlib import Path\nimport os\n",
            ),
        ],
    )
    text = QWEN35_FILE.read_text(encoding="utf-8")
    if "[PATCH] ISAAC advanced train scopes" not in text:
        text += r'''

# [PATCH] ISAAC advanced train scopes
_ISAAC_NATIVE_SETUP_QWEN35 = setup_qwen35_vla_for_training


def setup_qwen35_vla_for_training(*args, **kwargs):
    _ISAAC_NATIVE_SETUP_QWEN35(*args, **kwargs)
    scope = os.environ.get("ISAAC_TRAIN_SCOPE", "expert")
    if scope not in {"expert_connector", "expert_connector_lora"}:
        return
    model = args[0] if args else kwargs.get("model")
    if model is None:
        raise RuntimeError("ISAAC advanced training scope requires the model argument.")
    found = 0
    for name, parameter in model.named_parameters():
        if "model.vector_embedding." in name:
            parameter.requires_grad_(True)
            found += parameter.numel()
    if found == 0:
        raise RuntimeError("connector scope requested but model.vector_embedding parameters were not found")
'''
        QWEN35_FILE.write_text(text, encoding="utf-8")

    # Policy optimizer: keep connector trainable after PEFT and put it in its own LR group.
    _patch_text_once(
        ISAAC_POLICY_FILE,
        "[PATCH] ISAAC connector optimizer group",
        [
            (
                "import logging\nimport shutil\n",
                "import logging\nimport os\nimport shutil\n",
            ),
            (
                '                elif "lora_" in name:\n                    parameter.requires_grad_(True)\n        self._promote_trainable_parameters_to_fp32(self._isaac_model)\n        groups: dict[str, list[Tensor]] = {"vlm": [], "vision": [], "expert": []}\n',
                '                elif "lora_" in name:\n                    parameter.requires_grad_(True)\n                elif "vector_embedding." in name and os.environ.get("ISAAC_TRAIN_SCOPE") in {"expert_connector", "expert_connector_lora"}:\n                    parameter.requires_grad_(True)\n        self._promote_trainable_parameters_to_fp32(self._isaac_model)\n        groups: dict[str, list[Tensor]] = {"vlm": [], "vision": [], "expert": [], "connector": []}\n',
            ),
            (
                '            if "action_expert" in name:\n                groups["expert"].append(parameter)\n            elif name.startswith("model.visual."):\n',
                '            if "action_expert" in name:\n                groups["expert"].append(parameter)\n            elif "vector_embedding." in name:\n                groups["connector"].append(parameter)\n            elif name.startswith("model.visual."):\n',
            ),
            (
                '            ("expert", self.config.optimizer_action_expert_lr),\n        ):\n',
                '            ("expert", self.config.optimizer_action_expert_lr),\n            ("connector", float(os.environ.get("ISAAC_CONNECTOR_LR", self.config.optimizer_action_expert_lr))),\n        ):\n',
            ),
            (
                '            elif "lora_" in name:\n                parameter.requires_grad_(True)\n        self._promote_trainable_parameters_to_fp32(peft_model)\n',
                '            elif "lora_" in name:\n                parameter.requires_grad_(True)\n            elif "vector_embedding." in name and os.environ.get("ISAAC_TRAIN_SCOPE") in {"expert_connector", "expert_connector_lora"}:\n                parameter.requires_grad_(True)\n        self._promote_trainable_parameters_to_fp32(peft_model)\n',
            ),
        ],
    )


def _add_train_scope_args(cmd: list[str], scope: str, lora_r: int, lora_alpha: int, lora_dropout: float) -> list[str]:
    try:
        marker = cmd.index("-m")
        target = next(i for i in range(marker + 2, len(cmd) - 1) if cmd[i] == "-m" and cmd[i + 1] == "lerobot.scripts.lerobot_train")
    except (ValueError, StopIteration):
        raise RuntimeError("cannot locate lerobot_train in generated command")
    args = cmd[:target]
    policy_args = cmd[target:]
    if scope == "expert_connector_lora":
        # LeRobot's trainer creates cfg.peft when these nested fields are supplied.
        policy_args = [a for a in policy_args if a != "--policy.flow_matching_detach_vlm_activations=true"]
        policy_args += [
            "--policy.use_peft=true",
            "--peft.method_type=LORA",
            f"--peft.r={int(lora_r)}",
            f"--peft.lora_alpha={int(lora_alpha)}",
            f"--peft.lora_dropout={float(lora_dropout)}",
            '--peft.full_training_modules=["vector_embedding.0","vector_embedding.2"]',
        ]
    return args + policy_args


def _rewrite_backend(cmd: list[str], backend: str) -> tuple[list[str], bool]:
    launch_idx = cmd.index("-m")
    target_idx = next(i for i in range(launch_idx + 2, len(cmd) - 1) if cmd[i] == "-m" and cmd[i + 1] == "lerobot.scripts.lerobot_train")
    if backend == "fsdp1":
        return cmd, True
    if backend == "deepspeed3":
        cfg = Path(os.environ.get("ISAAC_DEEPSPEED_CONFIG", str(DEEPSPEED_CONFIG))).resolve()
        accel = [a for a in cmd[launch_idx + 2:target_idx] if not a.startswith("--mixed_precision=") and not a.startswith("--multi_gpu") and not a.startswith("--use_fsdp") and not a.startswith("--fsdp_") and not a.startswith("--dynamo_backend=")]
        accel += ["--use_deepspeed", f"--deepspeed_config_file={cfg}"]
        return cmd[:launch_idx + 2] + accel + cmd[target_idx:], False
    if backend == "fsdp2":
        cfg = Path(os.environ.get("ISAAC_FSDP2_CONFIG", str(FSDP2_CONFIG))).resolve()
        accel = [a for a in cmd[launch_idx + 2:target_idx] if not a.startswith("--mixed_precision=") and not a.startswith("--multi_gpu") and not a.startswith("--use_fsdp") and not a.startswith("--fsdp_") and not a.startswith("--dynamo_backend=")]
        accel += [f"--config_file={cfg}"]
        return cmd[:launch_idx + 2] + accel + cmd[target_idx:], False
    raise ValueError(f"unknown backend: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--train-scope", choices=["expert", "expert_connector", "expert_connector_lora"], default=os.environ.get("ISAAC_TRAIN_SCOPE", "expert_connector"))
    parser.add_argument("--backend", choices=["fsdp1", "fsdp2", "deepspeed3"], default=os.environ.get("ISAAC_BACKEND", "fsdp1"))
    parser.add_argument("--connector-lr", type=float, default=float(os.environ.get("ISAAC_CONNECTOR_LR", "1e-4")))
    parser.add_argument("--lora-r", type=int, default=int(os.environ.get("ISAAC_LORA_R", "64")))
    parser.add_argument("--lora-alpha", type=int, default=int(os.environ.get("ISAAC_LORA_ALPHA", "64")))
    parser.add_argument("--lora-dropout", type=float, default=float(os.environ.get("ISAAC_LORA_DROPOUT", "0.05")))
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    os.environ["ISAAC_TRAIN_SCOPE"] = known.train_scope
    os.environ["ISAAC_CONNECTOR_LR"] = str(known.connector_lr)
    apply_train_scope_patch()

    original_build = base.build_train_command
    def build_train_command(**kwargs):  # type: ignore[no-untyped-def]
        cmd = original_build(**kwargs)
        return _add_train_scope_args(cmd, known.train_scope, known.lora_r, known.lora_alpha, known.lora_dropout)
    base.build_train_command = build_train_command

    original_launch = base.launch_training
    def launch_training(cmd: list[str], **kwargs) -> None:  # type: ignore[no-untyped-def]
        rewritten, fsdp_flag = _rewrite_backend(cmd, known.backend)
        kwargs["fsdp"] = fsdp_flag
        if known.backend in {"fsdp2", "deepspeed3"}:
            os.environ["ISAAC_LOAD_TO_CPU"] = "1"
        print(f"[Advanced ISAAC] scope={known.train_scope} backend={known.backend} connector_lr={known.connector_lr}")
        if known.train_scope == "expert_connector_lora":
            print(f"[Advanced ISAAC] LoRA rank={known.lora_r} alpha={known.lora_alpha} dropout={known.lora_dropout}")
        return original_launch(rewritten, **kwargs)
    base.launch_training = launch_training

    base.main()


if __name__ == "__main__":
    main()
