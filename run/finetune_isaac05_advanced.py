#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified Isaac-0.5 quality-first launcher.

Train scopes:
  expert                 action expert only
  expert_connector       action expert + vector/proprio connector
  expert_connector_lora  action expert + connector + VLM/vision LoRA

Backends:
  fsdp1        existing stable Accelerate/FSDP path
  fsdp2        Accelerate FSDP2 config
  deepspeed3   DeepSpeed ZeRO-3 config

The wrapper reuses run/finetune_isaac05.py for dataset/package/model handling.
It only adds train-scope selection and the distributed backend switch.
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


def _patch_qwen_scope() -> None:
    marker = "# [PATCH] ISAAC advanced train scopes"
    text = QWEN35_FILE.read_text(encoding="utf-8")
    if marker in text:
        return
    if "from pathlib import Path\n" in text and "\nimport os\n" not in text:
        text = text.replace("from pathlib import Path\n", "from pathlib import Path\nimport os\n", 1)
    elif "import os\n" not in text:
        anchor = "from __future__ import annotations\n"
        if anchor not in text:
            raise RuntimeError(f"advanced patch anchor missing in {QWEN35_FILE}")
        text = text.replace(anchor, anchor + "\nimport os\n", 1)
    text += '''\n\n# [PATCH] ISAAC advanced train scopes\n# The native helper freezes everything except model.action_expert.* when\n# train_expert_only=True. Re-enable the vector/proprio connector for the\n# connector scopes after the native setup has established the safe baseline.\n_ISAAC_NATIVE_SETUP_QWEN35 = setup_qwen35_vla_for_training\n\n\ndef setup_qwen35_vla_for_training(*args, **kwargs):\n    _ISAAC_NATIVE_SETUP_QWEN35(*args, **kwargs)\n    scope = os.environ.get("ISAAC_TRAIN_SCOPE", "expert")\n    if scope not in {"expert_connector", "expert_connector_lora"}:\n        return\n    model = args[0] if args else kwargs.get("model")\n    if model is None:\n        raise RuntimeError("ISAAC advanced training scope requires the model argument")\n    trainable = 0\n    for name, parameter in model.named_parameters():\n        if "model.vector_embedding." in name:\n            parameter.requires_grad_(True)\n            trainable += parameter.numel()\n    if trainable == 0:\n        raise RuntimeError("connector scope requested but model.vector_embedding parameters were not found")\n'''
    QWEN35_FILE.write_text(text, encoding="utf-8")


def _patch_policy_scope() -> None:
    marker = "# [PATCH] ISAAC connector optimizer group"
    text = ISAAC_POLICY_FILE.read_text(encoding="utf-8")
    if marker in text:
        return
    original = text
    if "import logging\nimport shutil\n" not in text:
        raise RuntimeError("policy import anchor missing")
    text = text.replace("import logging\nimport shutil\n", "import logging\nimport os\nimport shutil\n", 1)

    old = '''        if self.config.use_peft:\n            # PEFT is applied in-place to this lazy inner model. Reassert the final\n            # contract here, immediately before optimizer groups are collected.\n            for name, parameter in self._isaac_model.named_parameters():\n                if "action_expert" in name:\n                    parameter.requires_grad_("modules_to_save.default" in name)\n                elif "lora_" in name:\n                    parameter.requires_grad_(True)\n        self._promote_trainable_parameters_to_fp32(self._isaac_model)\n        groups: dict[str, list[Tensor]] = {"vlm": [], "vision": [], "expert": []}\n'''
    new = '''        if self.config.use_peft:\n            # PEFT is applied in-place to this lazy inner model. Reassert the final\n            # contract here, immediately before optimizer groups are collected.\n            for name, parameter in self._isaac_model.named_parameters():\n                if "action_expert" in name:\n                    parameter.requires_grad_("modules_to_save.default" in name)\n                elif "lora_" in name:\n                    parameter.requires_grad_(True)\n                elif (\n                    "vector_embedding." in name\n                    and os.environ.get("ISAAC_TRAIN_SCOPE") in {"expert_connector", "expert_connector_lora"}\n                ):\n                    parameter.requires_grad_(True)\n        self._promote_trainable_parameters_to_fp32(self._isaac_model)\n        groups: dict[str, list[Tensor]] = {"vlm": [], "vision": [], "expert": [], "connector": []}\n'''
    if old not in text:
        raise RuntimeError("policy PEFT optimizer anchor missing")
    text = text.replace(old, new, 1)

    old = '''            if "action_expert" in name:\n                groups["expert"].append(parameter)\n            elif name.startswith("model.visual."):\n'''
    new = '''            if "action_expert" in name:\n                groups["expert"].append(parameter)\n            elif "vector_embedding." in name:\n                groups["connector"].append(parameter)\n            elif name.startswith("model.visual."):\n'''
    if old not in text:
        raise RuntimeError("policy optimizer grouping anchor missing")
    text = text.replace(old, new, 1)

    old = '''            ("expert", self.config.optimizer_action_expert_lr),\n        ):\n'''
    new = '''            ("expert", self.config.optimizer_action_expert_lr),\n            ("connector", float(os.environ.get("ISAAC_CONNECTOR_LR", self.config.optimizer_action_expert_lr))),\n        ):\n'''
    if old not in text:
        raise RuntimeError("policy LR grouping anchor missing")
    text = text.replace(old, new, 1)

    old = '''            elif "lora_" in name:\n                parameter.requires_grad_(True)\n        self._promote_trainable_parameters_to_fp32(peft_model)\n'''
    new = '''            elif "lora_" in name:\n                parameter.requires_grad_(True)\n            elif (\n                "vector_embedding." in name\n                and os.environ.get("ISAAC_TRAIN_SCOPE") in {"expert_connector", "expert_connector_lora"}\n            ):\n                parameter.requires_grad_(True)\n        self._promote_trainable_parameters_to_fp32(peft_model)\n'''
    if old not in text:
        raise RuntimeError("policy PEFT wrapper anchor missing")
    text = text.replace(old, new, 1)

    # Leave a marker at the modified optimizer method so future runs are no-ops.
    anchor = '    def get_optim_params(self) -> list[dict[str, Any]]:\n'
    if anchor not in text:
        raise RuntimeError("policy get_optim_params anchor missing")
    text = text.replace(anchor, anchor + f"        {marker}\n", 1)
    if text == original or marker not in text:
        raise RuntimeError("policy advanced patch verification failed")
    ISAAC_POLICY_FILE.write_text(text, encoding="utf-8")


def apply_train_scope_patch() -> None:
    if not QWEN35_FILE.is_file():
        raise FileNotFoundError(f"pinned LeRobot source not mounted: {QWEN35_FILE}")
    if not ISAAC_POLICY_FILE.is_file():
        raise FileNotFoundError(f"pinned LeRobot source not mounted: {ISAAC_POLICY_FILE}")
    _patch_qwen_scope()
    _patch_policy_scope()


def _add_train_scope_args(
    cmd: list[str], scope: str, lora_r: int, lora_alpha: int, lora_dropout: float
) -> list[str]:
    launch_idx = cmd.index("-m")
    target_idx = next(
        i for i in range(launch_idx + 2, len(cmd) - 1)
        if cmd[i] == "-m" and cmd[i + 1] == "lerobot.scripts.lerobot_train"
    )
    head = cmd[:target_idx]
    tail = cmd[target_idx:]
    if scope == "expert_connector_lora":
        tail = [a for a in tail if a != "--policy.flow_matching_detach_vlm_activations=true"]
        tail += [
            "--policy.use_peft=true",
            "--peft.method_type=LORA",
            f"--peft.r={int(lora_r)}",
            f"--peft.lora_alpha={int(lora_alpha)}",
            f"--peft.lora_dropout={float(lora_dropout)}",
            '--peft.full_training_modules=["vector_embedding.0","vector_embedding.2"]',
        ]
    return head + tail


def _rewrite_backend(cmd: list[str], backend: str) -> tuple[list[str], bool]:
    launch_idx = cmd.index("-m")
    target_idx = next(
        i for i in range(launch_idx + 2, len(cmd) - 1)
        if cmd[i] == "-m" and cmd[i + 1] == "lerobot.scripts.lerobot_train"
    )
    if backend == "fsdp1":
        return cmd, True

    accel = [
        a for a in cmd[launch_idx + 2 : target_idx]
        if not a.startswith("--mixed_precision=")
        and not a.startswith("--multi_gpu")
        and not a.startswith("--use_fsdp")
        and not a.startswith("--fsdp_")
        and not a.startswith("--dynamo_backend=")
    ]
    if backend == "deepspeed3":
        cfg = Path(os.environ.get("ISAAC_DEEPSPEED_CONFIG", str(DEEPSPEED_CONFIG))).resolve()
        accel += ["--use_deepspeed", f"--deepspeed_config_file={cfg}"]
        return cmd[: launch_idx + 2] + accel + cmd[target_idx:], False
    if backend == "fsdp2":
        cfg = Path(os.environ.get("ISAAC_FSDP2_CONFIG", str(FSDP2_CONFIG))).resolve()
        accel += [f"--config_file={cfg}"]
        return cmd[: launch_idx + 2] + accel + cmd[target_idx:], False
    raise ValueError(f"unknown backend: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--train-scope",
        choices=["expert", "expert_connector", "expert_connector_lora"],
        default=os.environ.get("ISAAC_TRAIN_SCOPE", "expert_connector"),
    )
    parser.add_argument(
        "--backend",
        choices=["fsdp1", "fsdp2", "deepspeed3"],
        default=os.environ.get("ISAAC_BACKEND", "fsdp1"),
    )
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
        print(
            f"[Advanced ISAAC] scope={known.train_scope} backend={known.backend} "
            f"connector_lr={known.connector_lr}"
        )
        if known.train_scope == "expert_connector_lora":
            print(
                f"[Advanced ISAAC] LoRA rank={known.lora_r} "
                f"alpha={known.lora_alpha} dropout={known.lora_dropout}"
            )
        return original_launch(rewritten, **kwargs)

    base.launch_training = launch_training
    base.main()


if __name__ == "__main__":
    main()
