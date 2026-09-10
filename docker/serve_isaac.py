"""Isaac 0.5 双臂 EEF 推理 WebSocket 服务端（参照 Enfold 服务端实现）。

协议（msgpack-numpy 编码）：
    请求: {
        "instruction": str,              # 任务指令（英文）
        "state": np.ndarray,             # [34] float32（与训练一致的 34D 状态）
        "images": {
            "head":  HxWx3 uint8,        # 相机键名与 patch 后的 camera_order 一致
            "left":  HxWx3 uint8,
            "right": HxWx3 uint8,
        }
    }
    响应: {
        "status": "success" | "error",
        "action": np.ndarray,            # [chunk_size, action_dim] 原始动作（反归一化）
        "latency": float,                # 毫秒
        "message": str,                  # 错误信息（失败时）
    }

用法:
    QWEN35_VOCAB_PATH=/vocab/vocab.json python serve_isaac.py \
        --policy-path /data/outputs/isaac-finetune/checkpoints/last/pretrained_model \
        --host 0.0.0.0 --port 8600
"""

import argparse
import asyncio
import time
from typing import Any

import msgpack_numpy as m
import numpy as np
import torch
import websockets

m.patch()

from lerobot.policies.perceptron_isaac import PerceptronIsaacPolicy
from lerobot.policies.perceptron_isaac.isaac_stats import unnormalize_isaac_actions


class IsaacServer:
    """Isaac 0.5 推理 websocket 服务。"""

    def __init__(self, policy: PerceptronIsaacPolicy, stateless: bool = True) -> None:
        self.policy = policy
        self.stateless = stateless
        self.action_dim = int(policy.config.action_dim)
        self.chunk_size = int(policy.config.chunk_size)
        self.cameras = list(policy.config.camera_order)
        self.proprio_dim = int(policy.config.proprio_dim)
        self.device = policy.device
        self.lock = asyncio.Lock()  # 显存锁，防止多并发崩溃
        print(
            f"服务就绪: action={self.action_dim}D, chunk={self.chunk_size}, "
            f"proprio={self.proprio_dim}D, cameras={self.cameras}, device={self.device}"
        )

    # ------------------------------------------------------------------ 预热
    def _warmup(self) -> None:
        print("正在进行模型预热...")
        dummy = {
            "observation.state": torch.zeros(1, self.proprio_dim, dtype=torch.float32),
            "task": "warmup",
        }
        for cam in self.cameras:
            dummy[f"observation.images.{cam}"] = torch.zeros(1, 240, 424, 3, dtype=torch.uint8)
        t0 = time.time()
        with torch.no_grad():
            self.policy.predict_action_chunk(dummy)
        print(f"预热完成，耗时 {time.time() - t0:.2f}s")

    # ------------------------------------------------------------- 推理（线程）
    def _run_inference(self, observation: dict[str, Any], instruction: str) -> np.ndarray:
        """在 executor 线程里跑推理，返回 [chunk_size, action_dim] 原始动作。"""
        batch: dict[str, Any] = {
            "observation.state": torch.from_numpy(
                np.asarray(observation["state"], dtype=np.float32)
            ).unsqueeze(0).to(self.device),
            "task": instruction,
        }
        for cam in self.cameras:
            img = np.asarray(observation["images"][cam])
            batch[f"observation.images.{cam}"] = (
                torch.from_numpy(img).unsqueeze(0).to(self.device)
            )

        if self.stateless:
            self.policy.reset()  # 每个请求独立（无历史泄漏）

        with torch.no_grad():
            action_norm = self.policy.predict_action_chunk(batch)  # [1, chunk, dim]
        raw = unnormalize_isaac_actions(action_norm.cpu().numpy(), self.policy._stats.action)
        return np.asarray(raw[0], dtype=np.float32)  # [chunk_size, action_dim]

    # ------------------------------------------------------------- websocket
    async def handler(self, websocket) -> None:
        print(f"新客户端连接: {websocket.remote_address}")
        try:
            async for message in websocket:
                data = m.unpackb(message)

                instruction = str(data.get("instruction", ""))
                images = data.get("images", {})
                observation = {
                    "state": np.asarray(data["state"], dtype=np.float32),
                    "images": {cam: np.asarray(images[cam]) for cam in self.cameras},
                }

                start_infer = time.time()
                async with self.lock:
                    loop = asyncio.get_event_loop()
                    action_chunk = await loop.run_in_executor(
                        None, self._run_inference, observation, instruction
                    )
                latency_ms = (time.time() - start_infer) * 1000
                print(f"模型推理耗时: {latency_ms / 1000:.2f}s")

                response = {
                    "status": "success",
                    "action": action_chunk,
                    "latency": latency_ms,
                }
                await websocket.send(m.packb(response))

        except websockets.exceptions.ConnectionClosed:
            print("客户端连接断开")
        except Exception as e:  # noqa: BLE001
            print(f"处理错误: {e}")
            try:
                await websocket.send(m.packb({"status": "error", "message": str(e)}))
            except Exception:
                pass


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Isaac 0.5 inference websocket server")
    parser.add_argument(
        "--policy-path",
        type=str,
        required=True,
        help="微调检查点目录 (checkpoints/last/pretrained_model)",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8600)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--stateless",
        type=lambda s: s.lower() in ("1", "true", "yes"),
        default=True,
        help="每个请求 reset 策略（无观测历史）；False = 保持滚动历史（闭环连续调用）",
    )
    return parser


async def main():
    args = _build_argparser().parse_args()
    print("policy-path:", args.policy_path)

    start_load = time.time()
    policy = PerceptronIsaacPolicy.from_pretrained(args.policy_path)
    policy.to(torch.device(args.device)).eval()
    print(f"策略加载耗时: {time.time() - start_load:.2f}s")

    server = IsaacServer(policy, stateless=args.stateless)
    server._warmup()
    print(f"模型加载耗时: {time.time() - start_load:.2f}s")

    async with websockets.serve(server.handler, args.host, args.port, max_size=None):
        print(f"WebSocket 服务运行在 ws://{args.host}:{args.port}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
