#!/usr/bin/env python
"""Smoke test for the official standalone mharmony integration (lerobot e12389c).

Exercises the full native pipeline exactly as LeRobot's Isaac path does:
    assert_mharmony_available -> load encoding -> build conversation ->
    render (image+text) -> rendered_stream_to_local_tensor_stream

Requires QWEN35_VOCAB_PATH (Qwen3.5 vocab.json) or an HF cache with the vocab.

Usage (inside the image):
    QWEN35_VOCAB_PATH=/vocab/vocab.json python docker/mharmony_smoke_test.py
"""
import argparse
import base64
import io
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
from PIL import Image

import mharmony
from lerobot.policies.perceptron_isaac.mharmony_adapter import (
    assert_mharmony_available,
    create_qwen35_image_processor,
    load_mharmony_encoding,
    rendered_stream_to_local_tensor_stream,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoding", default="QWEN35_HARMONY")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    print("== 1. assert_mharmony_available ==")
    assert_mharmony_available()
    print(f"   OK - standalone mharmony {getattr(mharmony, '__version__', '?')} "
          f"({mharmony.__file__})\n")

    print("== 2. encoding load ==")
    enc = load_mharmony_encoding(args.encoding)
    print(f"   encoding: {enc.name}\n")

    print("== 3. synthetic conversation (text + image) ==")
    img = Image.new("RGB", (64, 64), (120, 60, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    conv = mharmony.Conversation.from_messages(
        [
            mharmony.Message.from_role_and_contents(
                mharmony.Role.USER,
                [
                    mharmony.TextContent(text="What is in this image?"),
                    mharmony.ImageContent(
                        media_ref=mharmony.MediaRef(id="img0", mime="image/png", bytes_b64=b64)
                    ),
                ],
            ),
            mharmony.Message.from_role_and_content(
                mharmony.Role.ASSISTANT,
                mharmony.TextContent(text="It is a robot workspace."),
            ),
        ]
    )
    print(f"   conversation: {type(conv).__name__}, {len(conv.messages)} messages\n")

    print("== 4. render ==")
    proc = create_qwen35_image_processor(
        patch_size=16, max_num_patches=576, min_num_patches=None,
        pixel_shuffle_scale=1, temporal_patch_size=1,
    )
    preprocess_config = {
        "patch_size": 16, "max_num_patches": 576, "min_num_patches": None,
        "pixel_shuffle_scale": 1, "temporal_patch_size": 1,
    }
    rendered = enc.render_conversation_multimodal_with_processors(
        conv, preprocess_config=preprocess_config, image_processor=proc,
    )
    stream = rendered["stream"]
    print(f"   priority: {stream.get('priority')}")
    for i, ev in enumerate(stream.get("events", [])):
        print(f"   event[{i}] modality={ev.get('modality')} time={ev.get('time')} role={ev.get('role')}")

    print("\n== 5. rendered_stream_to_local_tensor_stream ==")
    ts = rendered_stream_to_local_tensor_stream(
        rendered, device=torch.device(args.device), dtype=torch.bfloat16
    )
    n = sum(1 for s in ts.streams for _ in s)
    print(f"   TensorStream OK: {len(ts.streams)} stream(s), {n} event(s)")
    for s in ts.streams:
        for ev in s:
            print(f"   event type={getattr(ev.type, 'name', ev.type)} (value={getattr(ev.type, 'value', '?')}) "
                  f"data.shape={tuple(ev.data.shape)} dtype={ev.data.dtype} dims_real={ev.dims_real}")

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
