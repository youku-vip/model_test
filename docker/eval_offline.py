#!/usr/bin/env python
"""离线测试：加载微调后的 Isaac 策略，在留出数据上预测动作并与真值对比。

无需真机/模拟器。会完整走一遍 mharmony 渲染 -> 流打包 -> 策略推理 -> 反归一化，
因此同时验证了微调效果和 shim 的正确性。

Usage (inside the isaac:gpu-mharmony image):
    QWEN35_VOCAB_PATH=/vocab/vocab.json python docker/eval_offline.py \
        --policy-path /data/outputs/isaac-finetune/checkpoints/last/pretrained_model \
        --dataset-repo-id local/isaac_joint14 \
        --dataset-root /data/lerobot-dataset \
        --n-frames 100 --smoke
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Any

# 优先使用挂载仓库的 lerobot（含 RTC 等补丁），避免用到镜像里的旧版 /lerobot/src/lerobot
_REPO_ROOT = Path(__file__).resolve().parents[1]
_LEROBOT_SRC = _REPO_ROOT / "lerobot" / "src"
if _LEROBOT_SRC.is_dir() and str(_LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(_LEROBOT_SRC))

# mharmony 需要 Qwen3.5 词表：默认用训练时的词表路径（可用环境变量覆盖）
os.environ.setdefault("QWEN35_VOCAB_PATH", "/data/algorithm/repo/vocab/qwen35/vocab.json")

import numpy as np
import torch

from lerobot.policies.perceptron_isaac import PerceptronIsaacPolicy
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

# ---------------------------------------------------------------------------
# A100(SM80) 推理豁免：模型远程代码 modeling_isaac05.py::require_qualified_runtime
# 强制 SM90(H100)。微调后推理是常规 BF16 前向，跳过 capability 检查。
# 方式：设环境变量 + 幂等文本补丁（transformers 会缓存 remote code，所以缓存/模型
# 目录的 modeling_isaac05.py 都打）。
# ---------------------------------------------------------------------------
_MODEL_GATE_MARKER = "# [PATCH] A100 expert-only fine-tuning"
_MODEL_GATE_ENV = "ISAAC_ALLOW_UNQUALIFIED_DEVICE"
_MODEL_GATE_ANCHOR = (
    '    if torch.version.cuda != _QUALIFIED_CUDA_VERSION:\n'
    '        installed_cuda = torch.version.cuda or "unavailable"\n'
    '        raise RuntimeError(\n'
    '            f"Isaac-0.5 CUDA inference requires CUDA {_QUALIFIED_CUDA_VERSION}, got {installed_cuda}."\n'
    '        )\n'
    '    capability = torch.cuda.get_device_capability(resolved_device)\n'
)
_MODEL_GATE_REPLACEMENT = (
    '    if torch.version.cuda != _QUALIFIED_CUDA_VERSION:\n'
    '        installed_cuda = torch.version.cuda or "unavailable"\n'
    '        raise RuntimeError(\n'
    '            f"Isaac-0.5 CUDA inference requires CUDA {_QUALIFIED_CUDA_VERSION}, got {installed_cuda}."\n'
    '        )\n'
    f'    if os.environ.get("{_MODEL_GATE_ENV}") == "1":\n'
    f'        # {_MODEL_GATE_MARKER}: 微调后常规 BF16 前向推理，跳过 SM90/H100 闸门\n'
    '        return\n'
    '    capability = torch.cuda.get_device_capability(resolved_device)\n'
)


def _allow_unqualified_device(gate_files: list[Path]) -> None:
    os.environ.setdefault(_MODEL_GATE_ENV, "1")
    for gate_file in gate_files:
        if not gate_file.is_file():
            continue
        try:
            text = gate_file.read_text(encoding="utf-8")
        except OSError:
            continue
        if _MODEL_GATE_MARKER in text:
            continue
        if _MODEL_GATE_ANCHOR not in text:
            continue
        text = text.replace(_MODEL_GATE_ANCHOR, _MODEL_GATE_REPLACEMENT, 1)
        if "import torch\nimport torch.nn.functional as F\n" in text:
            text = text.replace(
                "import torch\nimport torch.nn.functional as F\n",
                "import os\nimport torch\nimport torch.nn.functional as F\n",
                1,
            )
        gate_file.write_text(text, encoding="utf-8")
        print(f"[A100] 已给 {gate_file} 写入设备豁免（{_MODEL_GATE_ENV}=1 生效）")


def load_policy(path: str) -> PerceptronIsaacPolicy:
    # A100 推理豁免：打补丁的候选文件 = transformers 缓存 + checkpoint 内 hf_model + 模型目录
    candidates = [
        Path.home() / ".cache/huggingface/modules/transformers_modules/hf_model/modeling_isaac05.py",
        Path(path) / "hf_model" / "modeling_isaac05.py",
        Path(path).parent / "modeling_isaac05.py",
    ]
    _allow_unqualified_device(candidates)
    policy = PerceptronIsaacPolicy.from_pretrained(path)
    policy.eval()
    return policy


def build_observation(frame: dict, task: str, device: torch.device):
    """Convert a dataset frame into the policy batch dict (predict_action_chunk 需要 task 键)."""
    batch: dict[str, Any] = {}
    state = np.asarray(frame[OBS_STATE], dtype=np.float32)      # [dim]
    batch["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(device)  # [1, dim]
    batch["task"] = task
    for key, value in frame.items():
        if key.startswith(f"{OBS_IMAGES}."):
            img = np.ascontiguousarray(np.asarray(value))       # [H, W, C] uint8
            batch[key] = torch.from_numpy(img).unsqueeze(0).to(device)  # [1, H, W, C]
    return batch


def unnormalize_action(policy, action_norm: torch.Tensor) -> np.ndarray:
    """Invert the policy's normalization -> 原始动作 [chunk_size, action_dim]."""
    from lerobot.policies.perceptron_isaac.isaac_stats import unnormalize_isaac_actions

    raw = unnormalize_isaac_actions(action_norm.cpu().numpy(), policy._stats.action)
    return np.asarray(raw[0], dtype=np.float32)  # [chunk_size, action_dim]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-path", required=True, help="pretrained_model 目录")
    ap.add_argument("--dataset-repo-id", required=True)
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--n-frames", type=int, default=100)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--smoke", action="store_true", help="只跑 3 帧，验证链路")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"== 加载策略: {args.policy_path} (device={device}) ==")
    policy = load_policy(args.policy_path)

    print(f"== 加载数据集: {args.dataset_repo_id} ==")
    meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root)
    action_dim = int(meta.features["action"]["shape"][0])
    print(f"   action_dim={action_dim}, 共 {len(meta.episodes)} 个 episode")

    n = args.smoke and 3 or args.n_frames
    per_dim_abs_err = np.zeros(action_dim)
    preds, gts = [], []

    for ep in range(min(args.episodes, len(meta.episodes))):
        policy.reset()
        ep_meta = meta.episodes[ep]
        # 指令：优先 episode 自带的 tasks（描述文本），否则按 task_index 从 meta.tasks 取
        if ep_meta.get("tasks"):
            task = str(ep_meta["tasks"][0])
        else:
            task_idx = ep_meta.get("task", 0)
            tasks = getattr(meta, "tasks", None)
            if tasks is not None and len(tasks) > 0:
                task = str(tasks.index[task_idx] if task_idx < len(tasks) else tasks.index[0])
            else:
                task = ""
        # 全局帧起点（视频数据集存 dataset_from_index）
        from_idx = int(ep_meta.get("from_index", ep_meta.get("dataset_from_index", 0)))
        ep_len = ep_meta["length"]
        for fi in range(min(n, ep_len)):
            frame = dataset[from_idx + fi]
            batch = build_observation(frame, task, device)
            with torch.no_grad():
                action_norm = policy.predict_action_chunk(batch)   # [1, chunk, dim]
            raw = unnormalize_action(policy, action_norm)          # [chunk, dim]
            pred = raw[0]                                          # 立即动作（chunk 首行）
            gt = np.asarray(frame["action"], dtype=np.float32)[: action_dim]
            per_dim_abs_err += np.abs(pred - gt)
            preds.append(pred)
            gts.append(gt)

    total = min(args.episodes, len(meta.episodes)) * n
    mae = per_dim_abs_err / total
    print("\n== 结果（预测动作 vs 真值，绝对误差）==")
    names = ["x_l", "y_l", "z_l", "1d_l", "2d_l", "3d_l", "4d_l", "5d_l", "6d_l", "g_l",
             "x_r", "y_r", "z_r", "1d_r", "2d_r", "3d_r", "4d_r", "5d_r", "6d_r", "g_r"]
    for i in range(action_dim):
        print(f"   {names[i] if i < len(names) else i:>6}: MAE = {mae[i]:.4f}")
    print(f"\n   总帧数: {total} | 平均 MAE: {mae.mean():.4f}")
    if args.smoke:
        print("\n== SMOKE OK: 推理链路（含 mharmony 渲染/打包）正常 ==")


if __name__ == "__main__":
    main()
