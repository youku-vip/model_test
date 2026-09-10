#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""容器内环境探测：打印 GPU / 运行时版本 / 仓库挂载 / 模型包 / 数据集 / 词表 / 补丁状态。

在已有容器内先跑这个脚本确认路径，再按输出里的实际路径运行训练：
    python /code/run/probe_env.py

脚本只读，不修改任何文件；路径不确定时用它做一次性体检。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1]
LEROBOT_SRC = REPO_ROOT / "lerobot" / "src" / "lerobot"
PATCH_FILE = REPO_ROOT / "docker" / "patches" / "isaac05-expert-only-training.patch"
TRAIN_GATE_FILE = LEROBOT_SRC / "policies" / "perceptron_isaac" / "modeling_perceptron_isaac.py"
TRAIN_GATE_MARKER = "# [PATCH] Expert-only fine-tuning"
MODEL_GATE_MARKER = "# [PATCH] A100 expert-only fine-tuning"
MODEL_GATE_ENV = "ISAAC_ALLOW_UNQUALIFIED_DEVICE"

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
DATASET_CANDIDATES = (
    "/data/local/bimanual",
    "/data/ds",
    "/data/dataset",
)


def _ok(text: str) -> str:
    return f"  [OK]   {text}"


def _missing(text: str) -> str:
    return f"  [缺]   {text}"


def _warn(text: str) -> str:
    return f"  [注意] {text}"


def _probe_base_package(path: Path) -> Path | None:
    """校验一个路径是否为 Isaac-0.5 导入包（config.json type=perceptron_isaac）。"""
    cfg = path / "config.json"
    if not cfg.is_file():
        return None
    try:
        raw = json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raw = {}
    if raw.get("type") == "perceptron_isaac":
        print(_ok(f"{path}  (type={raw.get('type')}, action_dim={raw.get('action_dim')}, "
                  f"proprio_dim={raw.get('proprio_dim')}, cameras={raw.get('camera_order')})"))
        return path
    print(_warn(f"{path} 存在但 type={raw.get('type')!r}"))
    return None


def _probe_dataset(path: Path) -> Path | None:
    info = path / "meta" / "info.json"
    if not info.is_file():
        return None
    try:
        raw = json.loads(info.read_text(encoding="utf-8"))
        feats = raw.get("features", {})
        print(_ok(f"{path}  codebase={raw.get('codebase_version')} fps={raw.get('fps')} "
                  f"episodes={raw.get('total_episodes')} tasks={raw.get('total_tasks')}"))
        print(f"         state={feats.get('observation.state', {}).get('shape')} "
              f"action={feats.get('action', {}).get('shape')} "
              f"cameras={[k for k in feats if k.startswith('observation.images.')]}")
        return path
    except Exception as exc:  # noqa: BLE001
        print(_warn(f"{path} 读取 info.json 失败: {exc}"))
    return None


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Isaac 容器环境探测（只读）")
    parser.add_argument("--base-package", default=None, help="显式指定导入包路径并验证")
    parser.add_argument("--dataset-dir", default=None, help="显式指定数据集路径并验证")
    parser.add_argument("--vocab", default=None, help="显式指定词表路径并验证")
    args = parser.parse_args()

    print("=" * 70)
    print(" Isaac 容器环境探测（只读）")
    print("=" * 70)

    # ---- python / 运行时 ----
    print("\n== 运行时 ==")
    print(f"  python : {sys.version.split()[0]}")
    try:
        import torch

        print(f"  torch  : {torch.__version__} (cuda build {torch.version.cuda})")
        print(f"  cuda   : available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                cc = torch.cuda.get_device_capability(i)
                flag = "" if cc == (9, 0) else f"  <- A100 训练需 --allow-unqualified-device"
                print(f"    cuda:{i} {name} cc={cc}{flag}")
    except Exception as exc:  # noqa: BLE001
        print(_missing(f"torch 导入失败: {exc}"))
    try:
        import transformers

        print(f"  transformers : {transformers.__version__}  (便携模型要求 5.5.4)")
    except Exception as exc:  # noqa: BLE001
        print(_missing(f"transformers 导入失败: {exc}"))

    # ---- 仓库 / run 脚本 ----
    print("\n== 仓库挂载（run 脚本所在） ==")
    print(f"  本脚本   : {HERE}")
    print(f"  仓库根   : {REPO_ROOT}")
    print(f"  补丁文件 : {PATCH_FILE}")
    if PATCH_FILE.is_file():
        print(_ok(f"expert-only 训练补丁在仓库内: {PATCH_FILE}"))
    else:
        print(_missing("docker/patches/isaac05-expert-only-training.patch 不存在 —— 自动打补丁会失败"))

    # ---- lerobot 源码 + 补丁状态 ----
    print("\n== lerobot 源码 + PYTHONPATH ==")
    runtime_lerobot_src = None
    try:
        import lerobot  # noqa: F401

        print(f"  lerobot 包: {lerobot.__file__}")
        runtime_lerobot_src = Path(lerobot.__file__).resolve().parent  # .../lerobot/src/lerobot
    except Exception as exc:  # noqa: BLE001
        print(_warn(f"lerobot 导入失败（可能是 PYTHONPATH 问题）: {exc}"))
    mounted_lerobot_src = (REPO_ROOT / "lerobot" / "src" / "lerobot").resolve()
    print(f"  仓库 lerobot: {mounted_lerobot_src}")
    if runtime_lerobot_src is not None:
        if runtime_lerobot_src == mounted_lerobot_src:
            print(_ok("运行时导入的就是挂载仓库的 lerobot —— 自动补丁直接生效"))
        else:
            print(_warn(f"运行时导入的是 {runtime_lerobot_src}（镜像快照），不是挂载仓库！"))
            print(f"         自动补丁打在 {mounted_lerobot_src}，但训练实际加载 {runtime_lerobot_src} —— 补丁不会生效！")
            print(f"         修复：export PYTHONPATH={REPO_ROOT / 'lerobot' / 'src'}:$PYTHONPATH 后再跑训练")
            print("         （训练脚本也会自动在子进程 PYTHONPATH 里优先挂载仓库，见日志）")
    else:
        print(_missing("无法判断运行时 lerobot 来源（导入失败）"))
    if TRAIN_GATE_FILE.is_file():
        if TRAIN_GATE_MARKER in TRAIN_GATE_FILE.read_text(encoding="utf-8"):
            print(_ok(f"expert-only 训练补丁已应用: {TRAIN_GATE_FILE}"))
        else:
            print(_warn("expert-only 训练补丁未应用 —— 训练脚本会自动 git apply"))
            print(f"         手动: git -C {REPO_ROOT / 'lerobot'} apply {PATCH_FILE}")
    else:
        print(_missing(f"未找到 {TRAIN_GATE_FILE}（检查仓库挂载路径）"))

    # ---- 基础包（模型） ----
    print("\n== 基础包 / 模型 ==")
    found_package = None
    if args.base_package:
        found_package = _probe_base_package(Path(args.base_package))
    for candidate in BASE_PACKAGE_CANDIDATES:
        if found_package is not None and str(found_package) == candidate:
            continue
        found = _probe_base_package(Path(candidate))
        if found is not None:
            found_package = found
    if found_package is None:
        print(_missing("未定位到 Isaac-0.5 基础包"))
        print("         容器内查找：find / -maxdepth 6 -type d -name lerobot_policy 2>/dev/null")
        print("         然后: python probe_env.py --base-package <找到的路径>")
        for candidate in BASE_PACKAGE_CANDIDATES:
            print(f"         候选: {candidate}")

    # 模型远程代码 + A100 豁免状态
    if found_package is not None:
        model_root = found_package.parent
        gate_file = model_root / "modeling_isaac05.py"
        if gate_file.is_file():
            if MODEL_GATE_MARKER in gate_file.read_text(encoding="utf-8"):
                print(_ok(f"{MODEL_GATE_ENV}=1 豁免已写入: {gate_file}"))
            else:
                print(_warn(f"A100 豁免未写入: {gate_file} （训练加 --allow-unqualified-device 自动写）"))
        else:
            print(_warn(f"未找到模型远程代码 {gate_file}"))
        cfg_path = found_package / "config.json"
        try:
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
            hf = raw.get("hf_model_path")
            print(f"  包 hf_model_path={hf!r} -> 解析到 {found_package.parent}")
        except Exception:  # noqa: BLE001
            pass

    # ---- 数据集 ----
    print("\n== 数据集（meta/info.json） ==")
    found_dataset = None
    if args.dataset_dir:
        found_dataset = _probe_dataset(Path(args.dataset_dir))
    for candidate in DATASET_CANDIDATES:
        if found_dataset is not None and str(found_dataset) == candidate:
            continue
        found = _probe_dataset(Path(candidate))
        if found is not None:
            found_dataset = found
    if found_dataset is None:
        print(_missing("未定位到数据集"))
        print("         容器内查找：find / -maxdepth 6 -name info.json -path '*meta*' 2>/dev/null | head")
        print("         然后: python probe_env.py --dataset-dir <找到的目录>")
        for candidate in DATASET_CANDIDATES:
            print(f"         候选: {candidate}")

    # ---- 词表 ----
    print("\n== 词表 ==")
    vocab = None
    if args.vocab:
        if Path(args.vocab).is_file():
            vocab = args.vocab
            print(_ok(f"{vocab}"))
        else:
            print(_missing(f"--vocab 指定路径不存在: {args.vocab}"))
    for candidate in VOCAB_CANDIDATES:
        if Path(candidate).is_file():
            if vocab is None or vocab != candidate:
                vocab = candidate
                print(_ok(f"{candidate}"))
    if vocab is None:
        print(_missing("未定位到词表"))
        print("         容器内查找：find / -name vocab.json 2>/dev/null | head")
        print("         训练/部署需 QWEN35_VOCAB_PATH 或 --vocab")

    # ---- 可写性（自动补丁需要） ----
    print("\n== 可写性（自动补丁需要写挂载） ==")
    for label, path in (
        ("lerobot 源码", LEROBOT_SRC),
        ("模型目录", found_package.parent if found_package else None),
    ):
        if path is None:
            continue
        import os

        if os.access(path, os.W_OK):
            print(_ok(f"{label} 可写: {path}"))
        else:
            print(_warn(f"{label} 不可写: {path} —— 容器内 sudo chown -R user_lerobot:user_lerobot {path}"))

    print("\n" + "=" * 70)
    if found_package and found_dataset:
        print(" 探测完成，可直接运行：")
        print(f"   python {HERE.parent / 'finetune_isaac05.py'} \\")
        print(f"       --dataset-dir {found_dataset} --base-package {found_package} \\")
        print(f"       --gpus 0,1,2,3 --smoke --allow-unqualified-device")
    else:
        print(" 探测完成（部分路径未定位，训练时用 --dataset-dir / --base-package 显式指定）")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
