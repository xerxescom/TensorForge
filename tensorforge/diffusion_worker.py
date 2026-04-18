"""Subprocess worker for concurrent diffusion benchmark."""

from __future__ import annotations

import argparse
import json
import re
import time

import torch
from diffusers import AutoPipelineForText2Image

_MODEL_ALLOWED = re.compile(r"^[A-Za-z0-9._:/-]+$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TensorForge diffusion benchmark worker")
    parser.add_argument("--model", required=True, help="Diffusion model id")
    parser.add_argument("--duration", required=True, type=float, help="Benchmark duration in seconds")
    parser.add_argument(
        "--mode",
        default="cold_start",
        choices=["cold_start", "steady_state"],
        help="Measurement mode",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not _MODEL_ALLOWED.fullmatch(args.model):
        raise ValueError("invalid model id: only [A-Za-z0-9._:/-] are allowed")
    if args.duration <= 0:
        raise ValueError("duration must be > 0")


def main() -> int:
    args = _parse_args()
    _validate_args(args)

    n_images = 0
    total_steps = 0
    steps = 10

    load_t0 = time.perf_counter()
    pipe = AutoPipelineForText2Image.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")
    model_load_s = time.perf_counter() - load_t0

    warmup_s = 0.0
    if args.mode == "steady_state":
        warmup_t0 = time.perf_counter()
        pipe(prompt="warmup prompt", num_inference_steps=steps)
        warmup_s = time.perf_counter() - warmup_t0

    infer_t0 = time.perf_counter()
    t_end = time.time() + args.duration
    while time.time() < t_end:
        pipe(prompt="a red apple", num_inference_steps=steps)
        n_images += 1
        total_steps += steps

    inference_only_s = time.perf_counter() - infer_t0
    end_to_end_s = model_load_s + warmup_s + inference_only_s

    print(
        json.dumps(
            {
                "measurement_mode": args.mode,
                "model_load_s": round(model_load_s, 3),
                "warmup_s": round(warmup_s, 3),
                "inference_only_s": round(inference_only_s, 3),
                "end_to_end_s": round(end_to_end_s, 3),
                "n_images": n_images,
                "total_steps": total_steps,
                "it_per_s": round(total_steps / max(end_to_end_s, 1e-6), 3),
                "it_per_s_end_to_end": round(total_steps / max(end_to_end_s, 1e-6), 3),
                "it_per_s_inference_only": round(total_steps / max(inference_only_s, 1e-6), 3),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
