"""
Diffusion / CV / ASR benchmark tasks.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import ClassVar

from .collector import BenchmarkRunner, _subprocess_kwargs
from .model_manager import model_manager
from .tf_logger import logger


class DiffusionBenchmark(BenchmarkRunner):
    BENCH_PROMPTS: ClassVar[list[str]] = [
        "A futuristic city at night with neon lights reflecting on wet streets, cyberpunk style",
        "A serene mountain lake at sunrise, photorealistic, golden hour lighting",
        "A close-up portrait of a robot reading a book, detailed, cinematic",
        "Abstract geometric art, vivid colors, high contrast, minimalist",
        "A lush forest with rays of sunlight through the canopy, 8k",
    ]

    def __init__(
        self,
        model_name: str = "sdxl-turbo",
        precision: str = "fp16",
        n_images: int = 10,
        n_steps: int = 20,
        seed: int = 42,
        local_model_path: str | None = None,
        **kwargs,
    ):
        super().__init__(
            task_name="diffusion", model_name=model_name, precision=precision, **kwargs
        )
        self.n_images = n_images
        self.n_steps = n_steps
        self.seed = seed
        self.local_model_path = local_model_path
        self._ensure_model_available()

    def _ensure_model_available(self):
        if self.local_model_path:
            return
        try:
            model_path = model_manager.get_model_path(self.model_name)
            self.local_model_path = model_path
        except Exception as e:
            logger.warning(f"[diffusion] Model download failed: {e}")

    def run_task(self) -> dict:
        script = self._build_script()
        script_path = Path(self.output_dir) / "_diffusion_worker.py"
        script_path.write_text(script, "utf-8")

        t0 = time.perf_counter()
        try:
            out = subprocess.check_output(
                [sys.executable, str(script_path)],
                stderr=subprocess.STDOUT,
                timeout=900,
                **_subprocess_kwargs(),
            )
            elapsed = time.perf_counter() - t0
            lines = out.decode().strip().splitlines()
            worker_metrics = json.loads(lines[-1])
        except subprocess.CalledProcessError as e:
            elapsed = time.perf_counter() - t0
            worker_metrics = {
                "error": "process_failed",
                "details": (e.output or b"").decode(errors="ignore")[-500:],
            }
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - t0
            worker_metrics = {"error": "timeout"}
        except (FileNotFoundError, json.JSONDecodeError) as e:
            elapsed = time.perf_counter() - t0
            worker_metrics = {"error": "dependency_missing", "details": str(e)}

        total_steps = self.n_images * self.n_steps
        it_per_s = total_steps / elapsed if elapsed > 0 else 0
        seconds_per_image = elapsed / max(self.n_images, 1)

        return {
            "n_images": self.n_images,
            "n_steps": self.n_steps,
            "total_steps": total_steps,
            "total_elapsed_s": round(elapsed, 3),
            "seconds_per_image": round(seconds_per_image, 3),
            "it_per_s": round(it_per_s, 3),
            **worker_metrics,
        }

    def _build_script(self) -> str:
        dtype_map = {"fp16": "torch.float16", "fp32": "torch.float32", "bf16": "torch.bfloat16"}
        dtype = dtype_map.get(self.precision, "torch.float16")
        prompts_repr = repr(self.BENCH_PROMPTS)
        model_source = (
            f'"{self.local_model_path}"' if self.local_model_path else f'"{self.model_name}"'
        )

        cache_dir = str(model_manager.downloader.cache_dir)
        local_only = "True" if self.local_model_path else "False"

        return f"""
import json
import os
import sys
import time

import torch

os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
os.environ['TRANSFORMERS_CACHE'] = {cache_dir!r}
os.environ['HF_HOME'] = {cache_dir!r}

try:
    from diffusers import AutoPipelineForText2Image
except Exception as e:
    print(json.dumps({{"error": "diffusers_not_available", "details": str(e)}}))
    sys.exit(1)

pipe = AutoPipelineForText2Image.from_pretrained(
    {model_source},
    torch_dtype={dtype},
    variant="fp16",
    local_files_only={local_only},
    resume_download=True,
)
pipe = pipe.to("cuda")

prompts = {prompts_repr}
n = {self.n_images}
seed = {self.seed}
steps = {self.n_steps}
generator = torch.Generator("cuda").manual_seed(seed)

times = []
success_count = 0
for i in range(n):
    prompt = prompts[i % len(prompts)]
    t0 = time.perf_counter()
    try:
        _ = pipe(prompt=prompt, num_inference_steps=steps, generator=generator).images[0]
        times.append(time.perf_counter() - t0)
        success_count += 1
    except Exception:
        times.append(time.perf_counter() - t0)

if times:
    result = {{
        "per_image_s_mean": round(sum(times)/len(times), 3),
        "per_image_s_min": round(min(times), 3),
        "per_image_s_max": round(max(times), 3),
        "success_count": success_count,
        "total_attempts": n,
        "success_rate": round(success_count / n, 4),
    }}
else:
    result = {{"error": "no_successful_generations"}}

print(json.dumps(result))
"""


class CVBenchmark(BenchmarkRunner):
    def __init__(
        self,
        model_name: str = "yolov8n",
        precision: str = "fp16",
        n_frames: int = 200,
        image_size: int = 640,
        batch_size: int = 1,
        **kwargs,
    ):
        super().__init__(
            task_name="cv_detection", model_name=model_name, precision=precision, **kwargs
        )
        self.n_frames = n_frames
        self.image_size = image_size
        self.batch_size = max(int(batch_size), 1)

    def run_task(self) -> dict:
        script = self._build_script()
        script_path = Path(self.output_dir) / "_cv_worker.py"
        script_path.write_text(script, "utf-8")
        t0 = time.perf_counter()
        try:
            out = subprocess.check_output(
                [sys.executable, str(script_path)],
                stderr=subprocess.STDOUT,
                timeout=300,
                **_subprocess_kwargs(),
            )
            elapsed = time.perf_counter() - t0
            worker_metrics = json.loads(out.decode().strip().splitlines()[-1])
        except subprocess.CalledProcessError as e:
            elapsed = time.perf_counter() - t0
            worker_metrics = {
                "error": "process_failed",
                "details": (e.output or b"").decode(errors="ignore")[-800:],
            }
        except Exception as e:
            elapsed = time.perf_counter() - t0
            worker_metrics = {"error": type(e).__name__}

        return {
            "n_frames": self.n_frames,
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "total_elapsed_s": round(elapsed, 3),
            **worker_metrics,
        }

    def _build_script(self) -> str:
        half = "True" if self.precision == "fp16" else "False"
        return f"""
import json
import time

import numpy as np
import torch
from ultralytics import YOLO

half = {half}
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = YOLO("{self.model_name}.pt")
frame = np.zeros(({self.image_size}, {self.image_size}, 3), dtype=np.uint8)

for _ in range(10):
    warmup_batch = [frame] * {self.batch_size}
    model.predict(warmup_batch, imgsz={self.image_size}, device=device, half=(half and device.startswith("cuda")), verbose=False)

batch_latencies = []
per_frame_latencies = []
total_frames = {self.n_frames}
batch_size = {self.batch_size}
processed_batches = 0

for start in range(0, total_frames, batch_size):
    current_batch = min(batch_size, total_frames - start)
    inputs = [frame] * current_batch
    t0 = time.perf_counter()
    model.predict(inputs, imgsz={self.image_size}, device=device, half=(half and device.startswith("cuda")), verbose=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    batch_latencies.append(elapsed_ms)
    per_frame_latencies.append(elapsed_ms / current_batch)
    processed_batches += 1

lat = sorted(per_frame_latencies)
n = len(lat)
total_ms = sum(batch_latencies)

def pct_idx(total: int, q: float) -> int:
    return min(max(int(total * q), 0), total - 1)

print(json.dumps({{
    "device": device,
    "fps": round((total_frames * 1000) / total_ms, 2),
    "batch_per_s": round((processed_batches * 1000) / total_ms, 2),
    "latency_p50_ms": round(lat[pct_idx(n, 0.50)], 3),
    "latency_p95_ms": round(lat[pct_idx(n, 0.95)], 3),
    "latency_p99_ms": round(lat[pct_idx(n, 0.99)], 3),
    "latency_min_ms": round(lat[0], 3),
    "latency_max_ms": round(lat[-1], 3),
    "processed_batches": processed_batches,
}}))
"""


class ASRBenchmark(BenchmarkRunner):
    def __init__(
        self,
        model_name: str = "base",
        precision: str = "float16",
        audio_files: list[str] | None = None,
        ground_truths: list[str] | None = None,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(
            task_name="asr_whisper", model_name=model_name, precision=precision, **kwargs
        )
        self.audio_files = audio_files or []
        self.ground_truths = ground_truths or []
        self.device = device

    def run_task(self) -> dict:
        if not self.audio_files:
            return self._synthetic_benchmark()

        script = self._build_script()
        script_path = Path(self.output_dir) / "_asr_worker.py"
        script_path.write_text(script, "utf-8")
        try:
            out = subprocess.check_output(
                [sys.executable, str(script_path)],
                stderr=subprocess.STDOUT,
                timeout=600,
                **_subprocess_kwargs(),
            )
            return json.loads(out.decode().strip().splitlines()[-1])
        except Exception as e:
            logger.warning(f"  [warn] ASR worker failed: {e}")
            return {}

    def _synthetic_benchmark(self) -> dict:
        script = f"""
import json
import time

import numpy as np
from faster_whisper import WhisperModel

model = WhisperModel("{self.model_name}", device="{self.device}", compute_type="{self.precision}")
audio = np.random.randn(5 * 16000).astype(np.float32)
latencies = []
for _ in range(5):
    t0 = time.perf_counter()
    segs, _ = model.transcribe(audio, language="en")
    list(segs)
    latencies.append(time.perf_counter() - t0)

mean_lat = sum(latencies) / len(latencies)
print(json.dumps({{
    "rtf": round(mean_lat / 5.0, 4),
    "latency_mean_s": round(mean_lat, 3),
    "audio_duration_s": 5.0,
    "note": "synthetic_audio_benchmark",
}}))
"""
        p = Path(self.output_dir) / "_asr_synth.py"
        p.write_text(script, "utf-8")
        try:
            out = subprocess.check_output(
                [sys.executable, str(p)],
                stderr=subprocess.STDOUT,
                timeout=120,
                **_subprocess_kwargs(),
            )
            return json.loads(out.decode().strip().splitlines()[-1])
        except Exception as e:
            logger.warning(f"  [warn] Synthetic ASR failed: {e}")
            return {"note": "asr_not_available"}

    def _build_script(self) -> str:
        files_repr = repr(self.audio_files)
        truths_repr = repr(self.ground_truths)
        return f"""
import json
import time

import soundfile as sf
from difflib import SequenceMatcher
from faster_whisper import WhisperModel

model = WhisperModel("{self.model_name}", device="{self.device}", compute_type="{self.precision}")
audio_files = {files_repr}
ground_truths = {truths_repr}

results = []
for i, fpath in enumerate(audio_files):
    audio, sr = sf.read(fpath)
    duration_s = len(audio) / sr

    t0 = time.perf_counter()
    segs, info = model.transcribe(fpath, language="en")
    transcript = " ".join(s.text for s in segs)
    elapsed = time.perf_counter() - t0

    rtf = round(elapsed / duration_s, 4) if duration_s > 0 else 0
    wer = None
    if i < len(ground_truths):
        ref = ground_truths[i].lower().split()
        hyp = transcript.lower().split()
        sm = SequenceMatcher(None, ref, hyp)
        matches = sum(b.size for b in sm.get_matching_blocks())
        wer = round(1 - matches / max(len(ref), 1), 4)

    results.append({{
        "file": fpath,
        "rtf": rtf,
        "duration_s": round(duration_s, 2),
        "elapsed_s": round(elapsed, 3),
        "wer": wer,
    }})

rtf_list = [r["rtf"] for r in results]
wer_list = [r["wer"] for r in results if r["wer"] is not None]
print(json.dumps({{
    "n_files": len(results),
    "rtf_mean": round(sum(rtf_list)/len(rtf_list), 4) if rtf_list else 0,
    "rtf_min": round(min(rtf_list), 4) if rtf_list else 0,
    "wer_mean": round(sum(wer_list)/len(wer_list), 4) if wer_list else None,
    "per_file_detail": results,
}}))
"""
