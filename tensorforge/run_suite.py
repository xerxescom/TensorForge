"""
Full benchmark suite entrypoint and concurrent stress test.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .collector import GPUSampler, _subprocess_kwargs
from .config_manager import ConfigManager, apply_overrides
from .logging_utils import configure_logging
from .tf_logger import logger


class ConcurrentStressTest:
    def __init__(self, tasks: list[dict], output_dir: str = "results", gpu_index: int = 0):
        self.tasks = tasks
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_index = gpu_index

    def run(self) -> dict:
        logger.info(f"[Concurrent] Starting {len(self.tasks)} tasks simultaneously ...")
        sampler = GPUSampler(interval_s=0.5, gpu_index=self.gpu_index)
        baselines = self._run_baselines()

        sampler.start()
        t0 = time.time()
        threads: list[threading.Thread] = []
        concurrent_metrics: list[dict | None] = [None] * len(self.tasks)
        barrier = threading.Barrier(len(self.tasks))

        for i, task_cfg in enumerate(self.tasks):
            t = threading.Thread(
                target=self._run_one_task,
                args=(task_cfg, i, concurrent_metrics, barrier),
            )
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        elapsed = time.time() - t0
        samples = sampler.stop()
        gpu_stats = GPUSampler.summarize(samples)

        degradation: dict[str, float | None] = {}
        for cfg, conc in zip(self.tasks, concurrent_metrics, strict=True):
            if conc is None:
                continue
            key = f"{cfg['type']}_{cfg.get('model', '')}"
            baseline = baselines.get(key, {})
            for metric in ["tokens_per_s", "it_per_s", "fps"]:
                if metric in baseline and metric in conc:
                    degradation[f"{key}_{metric}_ratio"] = self._safe_ratio(
                        conc.get(metric), baseline.get(metric)
                    )

        result = {
            "test_type": "concurrent_stress",
            "n_tasks": len(self.tasks),
            "total_elapsed_s": round(elapsed, 3),
            "gpu_stats": gpu_stats,
            "baselines": baselines,
            "concurrent_metrics": concurrent_metrics,
            "degradation_ratios": degradation,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.output_dir / f"concurrent_{ts}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info(f"[Concurrent] Saved → {out_path.name}")
        return result

    @staticmethod
    def _safe_ratio(num, den):
        if num is None or den is None or den <= 0:
            return None
        return round(num / den, 4)

    def _run_one_task(self, cfg: dict, idx: int, results: list, barrier: threading.Barrier):
        t_type = cfg.get("type", "llm")
        duration = cfg.get("duration_s", 30)
        model = cfg.get("model", "")
        metrics: dict = {"type": t_type, "model": model}

        try:
            barrier.wait()
            if t_type == "llm":
                metrics.update(self._timed_llm(model, duration))
            elif t_type == "diffusion":
                metrics.update(self._timed_diffusion(model, duration))
        except Exception as e:
            metrics["error"] = str(e)

        results[idx] = metrics

    def _timed_llm(self, model: str, duration_s: float) -> dict:
        total_tokens = 0
        t_end = time.time() + duration_s
        prompt = "Explain quantum entanglement briefly."

        while time.time() < t_end:
            try:
                out = subprocess.run(
                    ["ollama", "run", model, prompt],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    encoding="utf-8",
                    **_subprocess_kwargs(),
                )
                total_tokens += int(len(out.stdout.split()) * 1.3)
            except Exception:
                break

        return {
            "tokens_generated": total_tokens,
            "tokens_per_s": round(total_tokens / duration_s, 2),
            "tokens_estimated": True,
        }

    def _timed_diffusion(self, model: str, duration_s: float) -> dict:
        # Keep this as a subprocess to isolate OOM, but avoid writing temp files.
        script = f"""
import json, time
import torch
from diffusers import AutoPipelineForText2Image

pipe = AutoPipelineForText2Image.from_pretrained(
    "{model}", torch_dtype=torch.float16, variant="fp16"
).to("cuda")

t_end = time.time() + {duration_s}
n_images = 0
total_steps = 0
steps = 10

while time.time() < t_end:
    pipe(prompt="a red apple", num_inference_steps=steps)
    n_images += 1
    total_steps += steps

print(json.dumps({{
    "n_images": n_images,
    "total_steps": total_steps,
    "it_per_s": round(total_steps / {duration_s}, 3),
}}))
"""
        try:
            out = subprocess.check_output(
                [sys.executable, "-c", script],
                timeout=duration_s + 30,
                **_subprocess_kwargs(),
            )
            return json.loads(out.decode().strip().splitlines()[-1])
        except Exception:
            return {}

    def _run_baselines(self) -> dict:
        logger.info("[Concurrent] Collecting single-task baselines ...")
        baselines: dict[str, dict] = {}
        for cfg in self.tasks:
            key = f"{cfg['type']}_{cfg.get('model', '')}"
            try:
                if cfg["type"] == "llm":
                    m = self._timed_llm(cfg.get("model", ""), cfg.get("duration_s", 20))
                elif cfg["type"] == "diffusion":
                    m = self._timed_diffusion(cfg.get("model", ""), cfg.get("duration_s", 20))
                else:
                    m = {}
                baselines[key] = m
            except Exception as e:
                baselines[key] = {"error": str(e)}
        return baselines


class FullBenchmarkSuite:
    def __init__(
        self,
        output_dir: str = "results",
        gpu_name: str = "GPU",
        gpu_index: int = 0,
        skip_phases: list[str] | None = None,
        config_path: str = "config.yaml",
        config_overrides: dict[str, str] | None = None,
        verbose: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_name = gpu_name
        self.gpu_index = gpu_index
        self.skip_phases = skip_phases or []
        self.all_results: list[dict] = []
        configure_logging(str(self.output_dir), verbose=verbose)
        cm = ConfigManager(config_path=config_path)
        cfg = cm.load_config()
        if config_overrides:
            cfg = apply_overrides(cfg, config_overrides)
        self.config = cfg
        self._write_resolved_config_snapshot(config_path, config_overrides or {})

    def _write_resolved_config_snapshot(self, config_path: str, overrides: dict[str, str]):
        snapshot = {
            "config_path": str(config_path),
            "overrides": overrides,
            "resolved": asdict(self.config),
        }
        (self.output_dir / "config_resolved.json").write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def run_all(self):
        logger.info("=" * 60)
        logger.info("  GPU Benchmark Suite")
        logger.info(f"  Target : {self.gpu_name}")
        logger.info(f"  Output : {self.output_dir}")
        logger.info(f"  Start  : {datetime.now().isoformat(timespec='seconds')}")
        logger.info("=" * 60)

        from .llm_bench import LLMBenchmark, LLMContextScaleBenchmark
        from .other_bench import ASRBenchmark, CVBenchmark, DiffusionBenchmark

        common: dict[str, Any] = dict(
            output_dir=str(self.output_dir),
            gpu_index=self.gpu_index,
            sample_interval_s=self.config.sample_interval_s,
            warmup_s=self.config.warmup_s,
            adaptive_sampling=self.config.adaptive_sampling,
            max_raw_samples=self.config.max_raw_samples,
        )

        phases = {
            "llm": lambda: LLMBenchmark(
                model_name=self.config.llm_model,
                precision=self.config.llm_precision,
                prompts=self.config.llm_prompts,
                n_runs=min(5, len(self.config.llm_prompts)),
                use_ollama=self.config.llm_backend == "ollama",
                **common,
            ).run(),
            "llm_fp16": lambda: LLMBenchmark(
                model_name=self.config.llm_model,
                precision=self.config.llm_fp16_precision,
                prompts=self.config.llm_prompts,
                n_runs=min(5, len(self.config.llm_prompts)),
                use_ollama=self.config.llm_backend == "ollama",
                **common,
            ).run(),
            "llm_context_scale": lambda: LLMContextScaleBenchmark(
                model_name=self.config.llm_model,
                precision=self.config.llm_precision,
                warmup_s=min(self.config.warmup_s, 3),
                **common,
            ).run(),
            "diffusion": lambda: DiffusionBenchmark(
                model_name=self.config.diffusion_model,
                precision=self.config.diffusion_precision,
                n_images=self.config.diffusion_n_images,
                n_steps=self.config.diffusion_n_steps,
                warmup_s=max(self.config.warmup_s, 10),
                **common,
            ).run(),
            "cv": lambda: CVBenchmark(
                model_name=self.config.cv_model,
                precision=self.config.cv_precision,
                n_frames=self.config.cv_n_frames,
                image_size=self.config.cv_image_size,
                **common,
            ).run(),
            "asr": lambda: ASRBenchmark(
                model_name=self.config.asr_model,
                precision=self.config.asr_precision,
                **common,
            ).run(),
            "concurrent": lambda: ConcurrentStressTest(
                tasks=[
                    {
                        "type": "llm",
                        "model": self.config.llm_model,
                        "duration_s": self.config.concurrent_duration_s,
                    },
                    {
                        "type": "diffusion",
                        "model": self.config.diffusion_model,
                        "duration_s": self.config.concurrent_duration_s,
                    },
                ],
                output_dir=str(self.output_dir),
                gpu_index=self.gpu_index,
            ).run(),
        }

        for phase_name, phase_fn in phases.items():
            if phase_name in self.skip_phases:
                logger.info(f"[Suite] Skipping phase: {phase_name}")
                continue
            logger.info(f"[Suite] ── Phase: {phase_name} ──")
            try:
                result = phase_fn()
                self.all_results.append({"phase": phase_name, "result": result})
            except Exception as e:
                logger.exception(f"[Suite] ERROR in {phase_name}: {e}")
                self.all_results.append({"phase": phase_name, "error": str(e)})

        self._write_final_report()

    def _write_final_report(self):
        report = {
            "gpu_name": self.gpu_name,
            "benchmark_date": datetime.now().isoformat(timespec="seconds"),
            "config_snapshot": asdict(self.config),
            "phases": self.all_results,
        }
        report_path = self.output_dir / "final_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, default=lambda o: o.__dict__), encoding="utf-8"
        )
        logger.info(f"[Suite] Final report → {report_path}")
        logger.info(f"[Suite] All done. Results in: {self.output_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GPU AI Benchmark Suite")
    parser.add_argument("--gpu-name", default="GPU", help="Friendly name for the GPU")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--set",
        dest="set_kv",
        action="append",
        default=[],
        help="Override config: key=value (key is a BenchmarkConfig field name)",
    )
    parser.add_argument(
        "--skip", nargs="*", default=[], help="Phase names to skip, e.g. --skip diffusion asr"
    )
    parser.add_argument(
        "--only", nargs="*", default=None, help="Run only these phases, e.g. --only llm cv"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose (debug) logging")
    args = parser.parse_args(argv)

    overrides: dict[str, str] = {}
    for item in args.set_kv or []:
        if "=" not in item:
            raise SystemExit(f"Invalid --set {item!r}. Expected key=value")
        k, v = item.split("=", 1)
        overrides[k.strip()] = v.strip()

    skip = args.skip
    if args.only:
        all_phases = [
            "llm",
            "llm_fp16",
            "llm_context_scale",
            "diffusion",
            "cv",
            "asr",
            "concurrent",
        ]
        skip = [p for p in all_phases if p not in args.only]

    suite = FullBenchmarkSuite(
        output_dir=args.output_dir,
        gpu_name=args.gpu_name,
        gpu_index=args.gpu_index,
        skip_phases=skip,
        config_path=args.config,
        config_overrides=overrides,
        verbose=args.verbose,
    )
    suite.run_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
