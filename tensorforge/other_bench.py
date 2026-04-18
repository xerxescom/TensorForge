"""
Diffusion / CV / ASR benchmark tasks.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from typing import ClassVar

from .collector import BenchmarkRunner, _subprocess_kwargs
from .error_schema import ensure_structured_error, structured_error
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
        t0 = time.perf_counter()
        try:
            cmd = self._build_worker_command()
            out = subprocess.check_output(
                cmd,
                stderr=subprocess.STDOUT,
                timeout=900,
                **_subprocess_kwargs(),
            )
            elapsed = time.perf_counter() - t0
            lines = out.decode().strip().splitlines()
            worker_metrics = json.loads(lines[-1])
        except subprocess.CalledProcessError as e:
            elapsed = time.perf_counter() - t0
            details = (e.output or b"").decode(errors="ignore")[-500:]
            worker_metrics = {
                "details": details,
                **structured_error(
                    error="process_failed",
                    error_stage="diffusion_worker_subprocess",
                    trace_text=details,
                ),
            }
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - t0
            worker_metrics = structured_error(
                error="timeout",
                error_type="timeout",
                error_stage="diffusion_worker_subprocess",
            )
        except (FileNotFoundError, json.JSONDecodeError) as e:
            elapsed = time.perf_counter() - t0
            worker_metrics = {
                "details": str(e),
                **structured_error(
                    error="dependency_missing",
                    error_type="dependency_missing",
                    error_stage="diffusion_worker_subprocess",
                    trace_text=str(e),
                ),
            }
        worker_metrics = ensure_structured_error(
            worker_metrics, error_stage="diffusion_worker_subprocess"
        )

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

    def _build_worker_command(self) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "tensorforge.diffusion_bench_worker",
            "--model",
            self.model_name,
            "--precision",
            self.precision,
            "--n-images",
            str(self.n_images),
            "--n-steps",
            str(self.n_steps),
            "--seed",
            str(self.seed),
            "--prompts-json",
            json.dumps(self.BENCH_PROMPTS, ensure_ascii=False),
            "--cache-dir",
            str(model_manager.downloader.cache_dir),
        ]
        if self.local_model_path:
            cmd.extend(["--local-model-path", self.local_model_path, "--local-files-only"])
        return cmd


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
        t0 = time.perf_counter()
        try:
            cmd = self._build_worker_command()
            out = subprocess.check_output(
                cmd,
                stderr=subprocess.STDOUT,
                timeout=300,
                **_subprocess_kwargs(),
            )
            elapsed = time.perf_counter() - t0
            worker_metrics = json.loads(out.decode().strip().splitlines()[-1])
        except subprocess.CalledProcessError as e:
            elapsed = time.perf_counter() - t0
            details = (e.output or b"").decode(errors="ignore")[-800:]
            worker_metrics = {
                "details": details,
                **structured_error(
                    error="process_failed",
                    error_stage="cv_worker_subprocess",
                    trace_text=details,
                ),
            }
        except Exception as e:
            elapsed = time.perf_counter() - t0
            worker_metrics = structured_error(
                error=type(e).__name__,
                error_stage="cv_worker_subprocess",
                trace_text=str(e),
            )
        worker_metrics = ensure_structured_error(worker_metrics, error_stage="cv_worker_subprocess")

        return {
            "n_frames": self.n_frames,
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "effective_batch_size": worker_metrics.get("effective_batch_size", self.batch_size),
            "total_elapsed_s": round(elapsed, 3),
            **worker_metrics,
        }

    def _build_worker_command(self) -> list[str]:
        return [
            sys.executable,
            "-m",
            "tensorforge.cv_bench_worker",
            "--model",
            self.model_name,
            "--precision",
            self.precision,
            "--n-frames",
            str(self.n_frames),
            "--image-size",
            str(self.image_size),
            "--batch-size",
            str(self.batch_size),
        ]


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

        try:
            cmd = self._build_worker_command(synthetic=False)
            out = subprocess.check_output(
                cmd,
                stderr=subprocess.STDOUT,
                timeout=600,
                **_subprocess_kwargs(),
            )
            return json.loads(out.decode().strip().splitlines()[-1])
        except Exception as e:
            logger.warning(f"  [warn] ASR worker failed: {e}")
            return structured_error(
                error=type(e).__name__,
                error_stage="asr_worker_subprocess",
                trace_text=str(e),
            )

    def _synthetic_benchmark(self) -> dict:
        try:
            cmd = self._build_worker_command(synthetic=True)
            out = subprocess.check_output(
                cmd,
                stderr=subprocess.STDOUT,
                timeout=120,
                **_subprocess_kwargs(),
            )
            return json.loads(out.decode().strip().splitlines()[-1])
        except Exception as e:
            logger.warning(f"  [warn] Synthetic ASR failed: {e}")
            return {
                "note": "asr_not_available",
                **structured_error(
                    error=type(e).__name__,
                    error_stage="asr_synthetic_subprocess",
                    trace_text=str(e),
                ),
            }

    def _build_worker_command(self, *, synthetic: bool) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "tensorforge.asr_bench_worker",
            "--model",
            self.model_name,
            "--precision",
            self.precision,
            "--device",
            self.device,
        ]
        if synthetic:
            cmd.append("--synthetic")
        else:
            cmd.extend(
                [
                    "--audio-files-json",
                    json.dumps(self.audio_files, ensure_ascii=False),
                    "--ground-truths-json",
                    json.dumps(self.ground_truths, ensure_ascii=False),
                ]
            )
        return cmd
