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
from .error_schema import classify_error_type, short_trace, structured_error
from .logging_utils import configure_logging
from .tf_logger import logger


class ConcurrentStressTest:
    """Run multiple workload types concurrently and compare against single-task baselines."""

    def __init__(
        self,
        tasks: list[dict],
        output_dir: str = "results",
        gpu_index: int = 0,
        measurement_mode: str = "cold_start",
        stats_precision_mode: str = "exact",
    ):
        self.tasks = tasks
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_index = gpu_index
        self.measurement_mode = (
            measurement_mode if measurement_mode in {"cold_start", "steady_state"} else "cold_start"
        )
        self.stats_precision_mode = (
            stats_precision_mode if stats_precision_mode in {"exact", "approximate"} else "exact"
        )

    def run(self) -> dict:
        logger.info(f"[Concurrent] Starting {len(self.tasks)} tasks simultaneously ...")
        sampler = GPUSampler(interval_s=0.5, gpu_index=self.gpu_index)
        # Phase 1: establish baseline throughput for each task type/model pair.
        baselines = self._run_baselines()

        # Phase 2: run all tasks under contention and collect shared GPU telemetry.
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
        gpu_stats = GPUSampler.summarize(
            samples,
            stats_precision_mode=self.stats_precision_mode,
        )

        # Phase 3: compute degradation ratios for comparable metrics.
        degradation: dict[str, float | None] = {}
        for cfg, conc in zip(self.tasks, concurrent_metrics, strict=True):
            if conc is None:
                continue
            key = f"{cfg['type']}_{cfg.get('model', '')}"
            baseline = baselines.get(key, {})
            for metric in [
                "tokens_per_s",
                "tokens_per_s_end_to_end",
                "tokens_per_s_inference_only",
                "it_per_s",
                "it_per_s_end_to_end",
                "it_per_s_inference_only",
                "fps",
            ]:
                if metric in baseline and metric in conc:
                    degradation[f"{key}_{metric}_ratio"] = self._safe_ratio(
                        conc.get(metric), baseline.get(metric)
                    )

        result = {
            "test_type": "concurrent_stress",
            "measurement_mode": self.measurement_mode,
            "stats_precision_mode": self.stats_precision_mode,
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
            metrics.update(
                structured_error(
                    error=e,
                    error_stage=f"concurrent_{t_type}_subprocess",
                    trace_text=str(e),
                )
            )

        results[idx] = metrics

    def _timed_llm(self, model: str, duration_s: float) -> dict:
        total_tokens = 0
        model_load_s = 0.0
        warmup_s = 0.0
        failed_rounds = 0
        successful_rounds = 0
        last_error_payload: dict[str, str] | None = None
        prompt = "Explain quantum entanglement briefly."
        warmup_prompt = "Reply with exactly one word: warm."
        end_to_end_t0 = time.perf_counter()

        if self.measurement_mode == "steady_state":
            try:
                load_t0 = time.perf_counter()
                subprocess.run(
                    ["ollama", "run", model, warmup_prompt],
                    capture_output=True,
                    text=True,
                    timeout=90,
                    encoding="utf-8",
                    check=True,
                    **_subprocess_kwargs(),
                )
                model_load_s = time.perf_counter() - load_t0

                warmup_t0 = time.perf_counter()
                subprocess.run(
                    ["ollama", "run", model, warmup_prompt],
                    capture_output=True,
                    text=True,
                    timeout=90,
                    encoding="utf-8",
                    check=True,
                    **_subprocess_kwargs(),
                )
                warmup_s = time.perf_counter() - warmup_t0
            except subprocess.CalledProcessError as e:
                err = structured_error(
                    error=e,
                    error_stage="concurrent_llm_warmup_subprocess",
                    trace_text=(e.stderr or e.stdout or str(e)),
                )
                return {
                    "measurement_mode": self.measurement_mode,
                    "tokens_generated": 0,
                    "tokens_per_s": 0.0,
                    "tokens_per_s_end_to_end": 0.0,
                    "tokens_per_s_inference_only": 0.0,
                    "tokens_estimated": True,
                    "model_load_s": 0.0,
                    "warmup_s": 0.0,
                    "inference_only_s": 0.0,
                    "end_to_end_s": round(time.perf_counter() - end_to_end_t0, 3),
                    "successful_rounds": 0,
                    "failed_rounds": 1,
                    "failure_rate": 1.0,
                    **err,
                }
            except Exception:
                model_load_s = 0.0
                warmup_s = 0.0

        t_infer_start = time.perf_counter()
        t_end = time.time() + duration_s

        while time.time() < t_end:
            try:
                one_t0 = time.perf_counter()
                out = subprocess.run(
                    ["ollama", "run", model, prompt],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    encoding="utf-8",
                    check=True,
                    **_subprocess_kwargs(),
                )
                if self.measurement_mode == "cold_start" and model_load_s == 0.0:
                    model_load_s = time.perf_counter() - one_t0
                total_tokens += int(len(out.stdout.split()) * 1.3)
                successful_rounds += 1
            except subprocess.CalledProcessError as e:
                failed_rounds += 1
                last_error_payload = structured_error(
                    error=e,
                    error_stage="concurrent_llm_inference_subprocess",
                    trace_text=(e.stderr or e.stdout or str(e)),
                )
                continue
            except Exception:
                failed_rounds += 1
                last_error_payload = structured_error(
                    error="llm_inference_exception",
                    error_stage="concurrent_llm_inference_subprocess",
                    trace_text="unexpected runtime exception in llm inference subprocess",
                )
                break

        inference_only_s = time.perf_counter() - t_infer_start
        end_to_end_s = time.perf_counter() - end_to_end_t0
        tokens_per_s_end_to_end = round(total_tokens / max(end_to_end_s, 1e-6), 2)
        tokens_per_s_inference_only = round(total_tokens / max(inference_only_s, 1e-6), 2)
        total_rounds = successful_rounds + failed_rounds
        failure_rate = round(failed_rounds / total_rounds, 4) if total_rounds > 0 else 0.0

        result = {
            "measurement_mode": self.measurement_mode,
            "tokens_generated": total_tokens,
            "tokens_per_s": tokens_per_s_end_to_end,
            "tokens_per_s_end_to_end": tokens_per_s_end_to_end,
            "tokens_per_s_inference_only": tokens_per_s_inference_only,
            "tokens_estimated": True,
            "model_load_s": round(model_load_s, 3),
            "warmup_s": round(warmup_s, 3),
            "inference_only_s": round(inference_only_s, 3),
            "end_to_end_s": round(end_to_end_s, 3),
            "successful_rounds": successful_rounds,
            "failed_rounds": failed_rounds,
            "failure_rate": failure_rate,
        }
        if last_error_payload:
            result.update(last_error_payload)
        return result

    def _timed_diffusion(self, model: str, duration_s: float) -> dict:
        # Keep this as a subprocess to isolate OOM, while using fixed script + CLI args.
        cmd = [
            sys.executable,
            "-m",
            "tensorforge.diffusion_worker",
            "--model",
            model,
            "--duration",
            str(duration_s),
            "--mode",
            self.measurement_mode,
        ]
        try:
            out = subprocess.check_output(
                cmd,
                timeout=duration_s + 30,
                **_subprocess_kwargs(),
            )
            return json.loads(out.decode().strip().splitlines()[-1])
        except Exception as e:
            return structured_error(
                error=type(e).__name__,
                error_stage="concurrent_diffusion_subprocess",
                trace_text=str(e),
            )

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
    """Orchestrate the full benchmark workflow and persist a final aggregate report."""

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
            stats_precision_mode=self.config.stats_precision_mode,
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
                batch_size=self.config.cv_batch_size,
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
                measurement_mode=self.config.concurrent_measurement_mode,
                stats_precision_mode=self.config.stats_precision_mode,
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
                self.all_results.append(
                    {
                        "phase": phase_name,
                        "error": str(e),
                        "error_type": classify_error_type(str(e)),
                        "error_stage": f"suite_phase_{phase_name}",
                        "short_trace": short_trace(str(e)),
                    }
                )

        self._write_final_report()

    @staticmethod
    def _collect_error_counts(payload: Any, counters: dict[str, int]):
        if isinstance(payload, dict):
            err = payload.get("error")
            if err:
                err_type = payload.get("error_type") or classify_error_type(str(err))
                if err_type in counters:
                    counters[err_type] += 1
            for value in payload.values():
                FullBenchmarkSuite._collect_error_counts(value, counters)
        elif isinstance(payload, list):
            for item in payload:
                FullBenchmarkSuite._collect_error_counts(item, counters)

    def _write_final_report(self):
        error_counts = {
            "download_failed": 0,
            "oom": 0,
            "dependency_missing": 0,
            "timeout": 0,
        }
        self._collect_error_counts(self.all_results, error_counts)
        report = {
            "gpu_name": self.gpu_name,
            "benchmark_date": datetime.now().isoformat(timespec="seconds"),
            "config_snapshot": asdict(self.config),
            "phases": self.all_results,
            "error_summary": error_counts,
        }
        report_path = self.output_dir / "final_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, default=lambda o: o.__dict__), encoding="utf-8"
        )
        logger.info(f"[Suite] Final report → {report_path}")
        logger.info(f"[Suite] All done. Results in: {self.output_dir}")


def main(argv: list[str] | None = None) -> int:
    def _load_structured_file(path: Path):
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            raise ValueError(f"file is empty: {path}")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            try:
                import yaml

                return yaml.safe_load(raw)
            except Exception as e:
                raise ValueError(
                    f"failed to parse {path} as JSON/YAML: {type(e).__name__}: {e}"
                ) from e

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
        help='Override config: key=value. Supports json: prefix, e.g. --set llm_prompts=json:["a","b"]',
    )
    parser.add_argument(
        "--set-file",
        dest="set_file_kv",
        action="append",
        default=[],
        help=(
            "Inject override from file. "
            "Use key=path (single field) or path to a JSON/YAML object with multiple keys."
        ),
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
    for item in args.set_file_kv or []:
        if "=" in item:
            k, path_raw = item.split("=", 1)
            key = k.strip()
        else:
            key = ""
            path_raw = item
        value_path = Path(path_raw.strip())
        if not value_path.exists():
            raise SystemExit(f"Invalid --set-file {item!r}. File not found: {value_path}")
        try:
            parsed = _load_structured_file(value_path)
        except Exception as e:
            raise SystemExit(f"Invalid --set-file {item!r}: {e}") from e
        if key:
            if isinstance(parsed, str):
                overrides[key] = parsed
            else:
                overrides[key] = f"json:{json.dumps(parsed, ensure_ascii=False)}"
        else:
            if not isinstance(parsed, dict):
                raise SystemExit(
                    f"Invalid --set-file {item!r}: expected JSON/YAML object when key is omitted"
                )
            for k, v in parsed.items():
                if isinstance(v, str):
                    overrides[str(k)] = v
                else:
                    overrides[str(k)] = f"json:{json.dumps(v, ensure_ascii=False)}"

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

    try:
        suite = FullBenchmarkSuite(
            output_dir=args.output_dir,
            gpu_name=args.gpu_name,
            gpu_index=args.gpu_index,
            skip_phases=skip,
            config_path=args.config,
            config_overrides=overrides,
            verbose=args.verbose,
        )
    except (KeyError, ValueError) as e:
        raise SystemExit(f"Config override error: {e}") from e
    suite.run_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
