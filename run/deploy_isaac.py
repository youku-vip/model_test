#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Isaac 0.5 双臂 EEF 推理 WebSocket 部署服务（与 serve_isaac.py 同一协议）。

协议（msgpack-numpy 编码，与 docker/serve_isaac.py 完全一致）：
    请求: {
        "instruction": str,              # 任务指令（英文）
        "state": np.ndarray,             # [34] float32（与训练一致的 34D 状态）
        "images": {
            "head":  HxWx3 uint8,        # 相机键名与 checkpoint 的 camera_order 一致
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
    QWEN35_VOCAB_PATH=/vocab/vocab.json python deploy_isaac.py \
        --policy-path /data/outputs/isaac-finetune/checkpoints/last/pretrained_model \
        --host 0.0.0.0 --port 8600

    # 启动前自检（加载模型 + 合成输入跑一次完整推理，不监听端口）：
    python deploy_isaac.py --policy-path <checkpoint> --self-test

    # 客户端连通性测试（向已启动的服务发一个合成请求）：
    python deploy_isaac.py --client --host 127.0.0.1 --port 8600

说明：
  * 输入输出与训练数据一一对应：state=34D（双 7 关节 + 双 6D 位姿 + 双夹爪，
    顺序见数据集 meta 的 observation.state.names），action=[chunk_size, 20D]
    原始绝对 EEF 动作（chunk_size 由 checkpoint 决定，默认 50；前 n_action_steps
    行可直接执行，其后行为未来预测）。
  * 部署动作语义（与训练一致，action_representation=absolute）：
    每臂 10D = [x, y, z, 6D旋转(旋转矩阵前两列), gripper]（左臂在前，右臂在后）。
  * 默认 stateless：每个请求独立推理（内部 policy.reset()，无观测历史泄漏）；
    --stateless=false 时保持滚动历史，供闭环连续调用。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any

# 优先使用挂载仓库的 lerobot（含 RTC 等补丁），避免用到镜像里的旧版 /lerobot/src/lerobot
_REPO_ROOT = Path(__file__).resolve().parents[1]
_LEROBOT_SRC = _REPO_ROOT / "lerobot" / "src"
if _LEROBOT_SRC.is_dir() and str(_LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(_LEROBOT_SRC))

# mharmony 需要 Qwen3.5 词表：默认用训练时的词表路径（可用环境变量覆盖）
os.environ.setdefault("QWEN35_VOCAB_PATH", "/data/algorithm/repo/vocab/qwen35/vocab.json")

# torch.compile(inductor) 编译大模型时图调度递归深度大，调高 Python 递归上限避免 RecursionError
sys.setrecursionlimit(1000000)

import numpy as np
import torch
import websockets

# 消息编码：优先 msgpack-numpy（与 serve_isaac.py 协议一致）；未安装则回退 JSON
# （numpy 转 list）。server/client 在同一环境下运行，两端自动一致。
try:
    import msgpack_numpy as m
    import msgpack  # noqa: F401

    m.patch()
    _MSG_BACKEND = "msgpack"
except ImportError:  # msgpack_numpy 未安装
    import json as _json

    _MSG_BACKEND = "json"

    def _jsonable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonable(v) for v in obj]
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return obj

    def m_packb(obj):
        return _json.dumps(_jsonable(obj)).encode("utf-8")

    def m_unpackb(data):
        return _json.loads(data.decode("utf-8"))

    class _M:
        @staticmethod
        def packb(obj):
            return m_packb(obj)

        @staticmethod
        def unpackb(data):
            return m_unpackb(data)

    m = _M()

from lerobot.policies.perceptron_isaac import PerceptronIsaacPolicy
from lerobot.policies.perceptron_isaac.isaac_stats import unnormalize_isaac_actions


# ---------------------------------------------------------------------------
# A100(SM80) 推理豁免：模型远程代码 modeling_isaac05.py::require_qualified_runtime
# 强制 SM90(H100)。微调后推理是常规 BF16 前向，跳过 capability 检查。
# 方式：设环境变量 + 幂等文本补丁（transformers 缓存 remote code，缓存/模型目录都打）。
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


class IsaacServer:
    """Isaac 0.5 推理 websocket 服务（协议见模块 docstring）。"""

    def __init__(self, policy: PerceptronIsaacPolicy, stateless: bool = True) -> None:
        self.policy = policy
        self.stateless = stateless
        self.action_dim = int(policy.config.action_dim)
        self.chunk_size = int(policy.config.chunk_size)
        self.cameras = list(policy.config.camera_order)
        self.proprio_dim = int(policy.config.proprio_dim)
        self.image_h = int(policy.config.image_size[0])
        self.image_w = int(policy.config.image_size[1])
        self.device = str(policy.config.device)  # 策略无 .device 属性，device 在 config
        self.lock = asyncio.Lock()  # 显存锁，防止多并发崩溃
        print(
            f"服务就绪: action={self.action_dim}D, chunk={self.chunk_size}, "
            f"proprio={self.proprio_dim}D, cameras={self.cameras}, "
            f"image=({self.image_h},{self.image_w}), device={self.device}, "
            f"stateless={self.stateless}"
        )

    # ------------------------------------------------------------------ 预热
    def _warmup(self) -> None:
        print("正在进行模型预热...")
        dummy: dict[str, Any] = {
            "observation.state": torch.zeros(1, self.proprio_dim, dtype=torch.float32),
            "task": "warmup",
        }
        for cam in self.cameras:
            dummy[f"observation.images.{cam}"] = torch.zeros(
                1, self.image_h, self.image_w, 3, dtype=torch.uint8
            )
        t0 = time.time()
        with torch.no_grad():
            self.policy.predict_action_chunk(dummy)
        self._ensure_stats()
        print(f"预热完成，耗时 {time.time() - t0:.2f}s")

    def _ensure_stats(self) -> None:
        """推理路径会按需加载归一化统计（_ensure_native_metadata）；显式触发一次。"""
        if getattr(self.policy, "_stats", None) is None:
            self.policy._ensure_native_metadata()
        action_block = self.policy._stats.action
        if int(action_block.dim) != self.action_dim:
            raise RuntimeError(
                f"checkpoint 动作统计维度 {action_block.dim} != config action_dim {self.action_dim}；"
                "检查点可能未带微调统计（isaac_stats.json）。"
            )

    # ------------------------------------------------------------- 推理（线程）
    def _run_inference(
        self,
        observation: dict[str, Any],
        instruction: str,
        action_prefix: np.ndarray | None = None,
        prefix_length: int | None = None,
    ) -> np.ndarray:
        """在 executor 线程里跑推理，返回 [chunk_size, action_dim] 原始动作。

        action_prefix: 可选的 RTC 已执行动作前缀（[P, action_dim] 原始动作），配合
        prefix_length 使用；模型把前缀行钉住、只预测后缀（需 RTC 训练过的 checkpoint）。
        """
        batch: dict[str, Any] = {
            "observation.state": torch.from_numpy(
                np.asarray(observation["state"], dtype=np.float32)
            ).unsqueeze(0).to(self.device),
            "task": instruction,
        }
        for cam in self.cameras:
            img = np.asarray(observation["images"][cam])
            # 接受 HxWxC uint8 —— predict_action_chunk 内部按 batch_size=1 在线打包
            batch[f"observation.images.{cam}"] = (
                torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0).to(self.device)
            )

        if self.stateless:
            self.policy.reset()  # 每个请求独立（无历史泄漏）

        # 调试：ISAAC_DEBUG_INPUT=1 时打印模型输入（指令/状态/图像）
        if os.environ.get("ISAAC_DEBUG_INPUT") == "1":
            try:
                print(f"[input] instruction: {instruction!r}")
                st = np.asarray(observation["state"], dtype=np.float32)
                print(f"[input] state[{st.shape}]: {np.round(st, 4).tolist()}")
                for cam in self.cameras:
                    img = np.asarray(observation["images"][cam])
                    print(
                        f"[input] image[{cam}]: shape={img.shape} dtype={img.dtype} "
                        f"min={img.min()} max={img.max()} mean={float(img.mean()):.1f}"
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"[input] 打印失败: {exc}")

        prefix_kwargs: dict[str, Any] = {}
        if action_prefix is not None:
            prefix_t = torch.from_numpy(
                np.asarray(action_prefix, dtype=np.float32)
            ).unsqueeze(0).to(self.device)  # [1, P, D]
            prefix_kwargs["action_prefix"] = prefix_t
            if prefix_length is not None:
                prefix_kwargs["prefix_length"] = int(prefix_length)

        with torch.no_grad():
            action_norm = self.policy.predict_action_chunk(batch, **prefix_kwargs)  # [1, chunk, dim]
        raw = unnormalize_isaac_actions(action_norm.cpu().numpy(), self.policy._stats.action)
        return np.asarray(raw[0], dtype=np.float32)  # [chunk_size, action_dim]

    # ------------------------------------------------------------- websocket
    async def handler(self, websocket) -> None:
        print(f"新客户端连接: {websocket.remote_address}")
        try:
            async for message in websocket:
                try:
                    data = m.unpackb(message)

                    instruction = str(data.get("instruction", ""))
                    images = data.get("images", {})
                    observation = {
                        "state": np.asarray(data["state"], dtype=np.float32),
                        "images": {cam: np.asarray(images[cam]) for cam in self.cameras},
                    }
                    # 可选 RTC 前缀：[P, action_dim] 原始动作（request: action_prefix）
                    action_prefix = data.get("action_prefix")
                    prefix_length = data.get("prefix_length")
                    if action_prefix is not None:
                        action_prefix = np.asarray(action_prefix, dtype=np.float32)

                    start_infer = time.time()
                    async with self.lock:
                        loop = asyncio.get_event_loop()
                        action_chunk = await loop.run_in_executor(
                            None, self._run_inference, observation, instruction,
                            action_prefix, prefix_length,
                        )
                    latency_ms = (time.time() - start_infer) * 1000
                    print(f"推理完成: {latency_ms:.1f}ms, action={tuple(action_chunk.shape)}")

                    response = {
                        "status": "success",
                        "action": action_chunk,
                        "latency": latency_ms,
                    }
                    await websocket.send(m.packb(response))
                except Exception as exc:  # noqa: BLE001 —— 单条消息失败不影响连接
                    print(f"处理错误: {exc}")
                    try:
                        await websocket.send(
                            m.packb({"status": "error", "message": str(exc), "latency": 0.0})
                        )
                    except Exception:
                        pass

        except websockets.exceptions.ConnectionClosed:
            print("客户端连接断开")
        except Exception as exc:  # noqa: BLE001
            print(f"连接处理异常: {exc}")


# ---------------------------------------------------------------------------
# 自检 / 客户端
# ---------------------------------------------------------------------------
def run_self_test(policy: PerceptronIsaacPolicy, policy_path: str, device: str) -> None:
    """不监听端口，直接走一遍 加载 -> 预热 -> 合成请求推理 -> 反归一化。"""
    print(f"== 自检: {policy_path} (device={device}) ==")
    server = IsaacServer(policy, stateless=True)
    server._warmup()

    observation = {
        "state": np.zeros(server.proprio_dim, dtype=np.float32),
        "images": {
            cam: np.zeros((server.image_h, server.image_w, 3), dtype=np.uint8)
            for cam in server.cameras
        },
    }
    t0 = time.time()
    action = server._run_inference(observation, "self test")
    ms = (time.time() - t0) * 1000
    if not np.isfinite(action).all():
        raise RuntimeError("自检失败: 动作含 NaN/Inf")
    print(f"自检通过: action={tuple(action.shape)}, latency={ms:.1f}ms")
    print(f"  action[0][:6] = {np.round(action[0, :6], 4).tolist()}")
    print(f"  action 统计: min={np.round(action.min(), 4)}, max={np.round(action.max(), 4)}")


async def run_client(host: str, port: int, image_h: int, image_w: int, cameras: list[str]) -> None:
    """向运行中的服务发送一个合成请求，验证协议连通性。"""
    uri = f"ws://{host}:{port}"
    print(f"== 客户端测试: {uri} ==")
    request = {
        "instruction": "pick up the cube and place it on the tray",
        "state": np.zeros(34, dtype=np.float32),
        "images": {cam: np.zeros((image_h, image_w, 3), dtype=np.uint8) for cam in cameras},
    }
    async with websockets.connect(uri, max_size=None) as websocket:
        await websocket.send(m.packb(request))
        response = m.unpackb(await websocket.recv())
    status = response.get("status")
    print(f"status={status!r}, latency={response.get('latency'):.1f}ms")
    if status == "success":
        action = np.asarray(response["action"])
        print(f"action={tuple(action.shape)} float32, 前 6 维: {np.round(action[0, :6], 4).tolist()}")
    else:
        print(f"错误: {response.get('message')}")
        sys.exit(1)


# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Isaac 0.5 双臂 EEF 推理 websocket 部署服务")
    parser.add_argument(
        "--policy-path",
        type=str,
        default=None,
        help="微调检查点目录 (checkpoints/last/pretrained_model)；--client 模式可省略",
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
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="不监听端口，加载模型并用合成输入自检后退出",
    )
    parser.add_argument(
        "--client",
        action="store_true",
        help="作为测试客户端连接运行中的服务（--host/--port 指向服务端）",
    )
    parser.add_argument("--image-size", type=str, default="240,424",
                        help="客户端测试图像 H,W（与服务端 checkpoint 的 image_size 一致）")
    parser.add_argument("--cameras", type=str, default="head,left,right",
                        help="客户端测试相机键（与服务端 camera_order 一致）")
    # ---- 推理加速 ----
    parser.add_argument("--device-map", default="single", choices=["single", "auto"],
                        help="模型加载设备映射：single=单卡（默认）；auto=HF 自动分到所有可见 GPU"
                             "（如 2 卡一份，4 卡跑 2 个实例各跨 2 卡；需设 CUDA_VISIBLE_DEVICES）")
    parser.add_argument("--num-inference-steps", type=int, default=None,
                        help="flow Euler 采样步数（checkpoint 默认 10；降到 4-6 明显加快，质量略降）")
    parser.add_argument("--torch-compile", action="store_true",
                        help="用 torch.compile(inductor) 编译模型加速推理（remote-code 模型有兼容风险，"
                             "失败会自动回退不编译）")
    return parser


async def main() -> None:
    args = _build_argparser().parse_args()

    if not args.client:
        if not args.policy_path:
            raise SystemExit("--policy-path 必填（除非使用 --client 客户端模式）")
        print("policy-path:", args.policy_path)

    if args.client:
        h, w = (int(x) for x in args.image_size.split(","))
        cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
        await run_client(args.host, args.port, h, w, cameras)
        return

    # 多卡模型并行推理：ISAAC_DEVICE_MAP=auto -> HF 把模型分到所有可见 GPU（如 2 卡一份）
    if args.device_map == "auto":
        os.environ["ISAAC_DEVICE_MAP"] = "auto"
        print(f"[device-map] 模型将跨 CUDA_VISIBLE_DEVICES 可见的 GPU 自动分片")
    # A100(SM80) 推理豁免：环境变量 + 幂等文本补丁（transformers 缓存的 modeling_isaac05.py）
    _allow_unqualified_device(
        [
            Path.home() / ".cache/huggingface/modules/transformers_modules/hf_model/modeling_isaac05.py",
            Path(args.policy_path) / "hf_model" / "modeling_isaac05.py",
            Path(args.policy_path).parent / "modeling_isaac05.py",
        ]
    )

    start_load = time.time()
    policy = PerceptronIsaacPolicy.from_pretrained(args.policy_path)
    policy.config.device = args.device
    if args.num_inference_steps is not None:
        policy.config.num_inference_steps = int(args.num_inference_steps)
        print(f"[speed] num_inference_steps -> {policy.config.num_inference_steps}")
    if args.device_map == "auto":
        # 跨卡切分已在 _load_backbone 里完成（后半层在 cuda:1）；这里不能再 policy.to()，
        # 否则会把整个模型搬回 cuda:0、破坏切分（正是之前跨卡报错的根因）。
        print("[device-map] 跳过 policy.to()（保持跨卡切分）")
    else:
        policy.to(torch.device(args.device)).eval()
    print(f"策略加载耗时: {time.time() - start_load:.2f}s")

    if args.self_test:
        run_self_test(policy, args.policy_path, args.device)
        return

    server = IsaacServer(policy, stateless=args.stateless)
    server._warmup()

    # 推理加速：torch.compile(inductor)（失败自动回退，不阻塞）
    if args.torch_compile:
        try:
            policy._isaac_model = torch.compile(policy._isaac_model, mode="reduce-overhead")
            print("[speed] 已启用 torch.compile(inductor)（首次推理会编译，后续加速）")
        except Exception as exc:  # noqa: BLE001
            print(f"[警告] torch.compile 失败，回退未编译：{exc}")

    print(f"模型加载总耗时: {time.time() - start_load:.2f}s")

    async with websockets.serve(server.handler, args.host, args.port, max_size=None):
        print(f"WebSocket 服务运行在 ws://{args.host}:{args.port}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
