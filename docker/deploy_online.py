#!/usr/bin/env python
"""在线部署脚本：加载微调后的 Isaac-0.5 双臂策略，闭环推理并输出机器人指令。

训练产物 -> 真机控制的桥梁。你需要实现两个回调：
  get_observation() -> dict{observation.state: [34], observation.images.<cam>: [H,W,C]uint8, ...}
  execute_action(raw20) -> 把 20D 动作转成睿尔曼指令并发送（movel_p / 夹爪）

动作语义（与训练数据一致，action_representation=absolute）：
  每臂 10D = [x, y, z, 6D旋转(旋转矩阵前两列), gripper]（左臂在前，右臂在后）
  部署层把 6D 旋转还原成旋转矩阵 -> 欧拉角 -> 睿尔曼 SDK 位姿指令。

用法：
  QWEN35_VOCAB_PATH=/vocab/vocab.json python deploy_online.py \
      --policy-path /data/outputs/isaac-finetune/checkpoints/last/pretrained_model \
      --fps 10 --cameras head,left,right
"""
import argparse
import time

import numpy as np
import torch

from lerobot.policies.perceptron_isaac import PerceptronIsaacPolicy
from lerobot.policies.perceptron_isaac.isaac_stats import unnormalize_isaac_actions


# ---------------------------------------------------------------------------
# ① 观测回调：从你的相机/机器人读数据（按实际硬件实现）
# ---------------------------------------------------------------------------
def get_observation(cameras: list[str]) -> dict:
    """返回 batch 输入。示例：从相机采集 + 读睿尔曼状态。

    返回:
        {
          "observation.state": np.ndarray [34],   # 关节14 + xyz6 + rot6d*2 + gripper2（与训练一致）
          "observation.images.head":  np.ndarray [H,W,C] uint8,
          "observation.images.left":  ...,
          "observation.images.right": ...,
        }
    """
    raise NotImplementedError(
        "实现 get_observation()：采集 3 路相机 + 34D 机器人状态。"
        "可用 RealMan SDK (rm_api) 读关节/位姿，相机用 OpenCV/相机 SDK。"
    )


# ---------------------------------------------------------------------------
# ② 动作执行回调：20D 动作 -> 睿尔曼指令（按实际硬件实现）
# ---------------------------------------------------------------------------
def rot6d_to_matrix(r6):
    r1 = np.asarray(r6[:3], float); r1 = r1 / np.linalg.norm(r1)
    r2 = np.asarray(r6[3:6], float)
    r2 = r2 - np.dot(r1, r2) * r1; r2 = r2 / np.linalg.norm(r2)
    return np.stack([r1, r2, np.cross(r1, r2)], axis=1)


def matrix_to_euler(R):
    sy = np.hypot(R[0, 0], R[1, 0])
    if sy > 1e-8:
        return np.array([np.arctan2(R[2, 1], R[2, 2]),
                         np.arctan2(-R[2, 0], sy),
                         np.arctan2(R[1, 0], R[0, 0])])
    return np.zeros(3)


def execute_action(raw_action: np.ndarray) -> None:
    """raw_action: [20] = 左臂10D + 右臂10D (absolute EEF + gripper)。"""
    left, right = raw_action[:10], raw_action[10:]
    for name, arm in (("left", left), ("right", right)):
        xyz = arm[0:3]
        R = rot6d_to_matrix(arm[3:9])
        rpy = matrix_to_euler(R)
        gripper = float(arm[9])
        print(f"  [{name}] pos={xyz} rpy={rpy} gripper={gripper:.2f}")
        # TODO: 用睿尔曼 SDK 发送（如 rm_api 的 movel_p / 夹爪开合）
        #   rm_movep(robot_handle, xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2], ...)
        #   rm_set_gripper_pick_on / off 或夹爪开度指令
    raise NotImplementedError("实现 execute_action()：调用睿尔曼 SDK 发送左/右臂指令")


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-path", required=True)
    ap.add_argument("--cameras", default="head,left,right")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-steps", type=int, default=0, help="0=无限运行")
    args = ap.parse_args()

    device = torch.device(args.device)
    cameras = [c.strip() for c in args.cameras.split(",")]
    print(f"== 加载策略: {args.policy_path} ==")
    policy = PerceptronIsaacPolicy.from_pretrained(args.policy_path)
    policy.to(device).eval()

    policy.reset()
    period = 1.0 / args.fps
    step = 0
    print(f"== 开始闭环推理 @ {args.fps}fps ==")
    try:
        while True:
            t0 = time.perf_counter()
            obs = get_observation(cameras)
            batch = {"observation.state": torch.from_numpy(
                np.asarray(obs["observation.state"], np.float32)).unsqueeze(0).to(device)}
            for cam in cameras:
                img = np.asarray(obs[f"observation.images.{cam}"])
                batch[f"observation.images.{cam}"] = torch.from_numpy(img).unsqueeze(0).to(device)

            with torch.no_grad():
                action_norm = policy.select_action(batch)      # [1, action_dim] 归一化
            raw = unnormalize_isaac_actions(action_norm.cpu().numpy(), policy._stats.action)
            execute_action(raw[0])

            step += 1
            if args.max_steps and step >= args.max_steps:
                break
            elapsed = time.perf_counter() - t0
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        print("\n== 停止 ==")


if __name__ == "__main__":
    main()
