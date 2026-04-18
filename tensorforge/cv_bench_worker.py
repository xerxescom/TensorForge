"""Subprocess worker for CVBenchmark."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from ultralytics import YOLO


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TensorForge CV benchmark worker")
    parser.add_argument("--model", required=True, help="YOLO model name without .pt")
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--n-frames", type=int, required=True)
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    return parser.parse_args()


def _pct_idx(total: int, q: float) -> int:
    return min(max(int(total * q), 0), total - 1)


def main() -> int:
    args = _parse_args()
    half = args.precision == "fp16"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = YOLO(f"{args.model}.pt")
    frame = np.zeros((args.image_size, args.image_size, 3), dtype=np.uint8)

    for _ in range(10):
        warmup_batch = [frame] * args.batch_size
        model.predict(
            warmup_batch,
            imgsz=args.image_size,
            device=device,
            half=(half and device.startswith("cuda")),
            verbose=False,
        )

    batch_latencies: list[float] = []
    per_frame_latencies: list[float] = []
    processed_batches = 0

    for start in range(0, args.n_frames, args.batch_size):
        current_batch = min(args.batch_size, args.n_frames - start)
        inputs = [frame] * current_batch
        t0 = time.perf_counter()
        model.predict(
            inputs,
            imgsz=args.image_size,
            device=device,
            half=(half and device.startswith("cuda")),
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        batch_latencies.append(elapsed_ms)
        per_frame_latencies.append(elapsed_ms / current_batch)
        processed_batches += 1

    lat = sorted(per_frame_latencies)
    n = len(lat)
    total_ms = sum(batch_latencies)

    print(
        json.dumps(
            {
                "device": device,
                "configured_batch_size": args.batch_size,
                "effective_batch_size": round(args.n_frames / max(processed_batches, 1), 3),
                "fps": round((args.n_frames * 1000) / total_ms, 2),
                "batch_per_s": round((processed_batches * 1000) / total_ms, 2),
                "latency_p50_ms": round(lat[_pct_idx(n, 0.50)], 3),
                "latency_p95_ms": round(lat[_pct_idx(n, 0.95)], 3),
                "latency_p99_ms": round(lat[_pct_idx(n, 0.99)], 3),
                "latency_min_ms": round(lat[0], 3),
                "latency_max_ms": round(lat[-1], 3),
                "processed_batches": processed_batches,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
