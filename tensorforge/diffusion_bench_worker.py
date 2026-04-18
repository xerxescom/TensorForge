"""Subprocess worker for DiffusionBenchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TensorForge diffusion benchmark worker")
    parser.add_argument("--model", required=True, help="Diffusion model id")
    parser.add_argument("--local-model-path", default="", help="Local model path")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "fp32", "bf16"])
    parser.add_argument("--n-images", type=int, required=True)
    parser.add_argument("--n-steps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompts-json", required=True, help="JSON array of prompts")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _dtype_for_precision(precision: str):
    return {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }.get(precision, torch.float16)


def main() -> int:
    args = _parse_args()

    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["TRANSFORMERS_CACHE"] = args.cache_dir
    os.environ["HF_HOME"] = args.cache_dir

    try:
        from diffusers import AutoPipelineForText2Image
    except Exception as e:  # pragma: no cover - dependency not always present
        print(json.dumps({"error": "diffusers_not_available", "details": str(e)}))
        return 1

    prompts = json.loads(args.prompts_json)
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompts-json must be a non-empty JSON array")

    model_source = args.local_model_path or args.model
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_source,
        torch_dtype=_dtype_for_precision(args.precision),
        variant="fp16",
        local_files_only=args.local_files_only,
        resume_download=True,
    )
    pipe = pipe.to("cuda")

    generator = torch.Generator("cuda").manual_seed(args.seed)
    times: list[float] = []
    success_count = 0
    for i in range(args.n_images):
        prompt = str(prompts[i % len(prompts)])
        t0 = time.perf_counter()
        try:
            _ = pipe(prompt=prompt, num_inference_steps=args.n_steps, generator=generator).images[0]
            times.append(time.perf_counter() - t0)
            success_count += 1
        except Exception:
            times.append(time.perf_counter() - t0)

    if not times:
        print(json.dumps({"error": "no_successful_generations"}))
        return 0

    result = {
        "per_image_s_mean": round(sum(times) / len(times), 3),
        "per_image_s_min": round(min(times), 3),
        "per_image_s_max": round(max(times), 3),
        "success_count": success_count,
        "total_attempts": args.n_images,
        "success_rate": round(success_count / max(args.n_images, 1), 4),
    }
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
