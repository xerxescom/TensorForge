"""
多任务并发压测 + 完整测试套件入口
支持平台：Windows 10/11 · Linux
"""
import argparse
import time
import json
import sys
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

from ..core.tf_logger import logger

from ..core.collector import GPUSampler, _subprocess_kwargs
from ..core.config_manager import config_manager
from ..core.logging_utils import configure_logging


# ─────────────────────────────────────────────────────────────
#  并发压测
# ─────────────────────────────────────────────────────────────
class ConcurrentStressTest:
    """
    同时运行多个推理任务，测量:
      - 各任务吞吐量在并发时的衰减率
      - VRAM 峰值与 OOM 边界
      - 功耗在混合负载下的行为
    """

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
        threads = []
        concurrent_metrics = [None] * len(self.tasks)
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

        degradation = {}
        for cfg, conc in zip(self.tasks, concurrent_metrics):
            key = f"{cfg['type']}_{cfg.get('model', '')}"
            baseline = baselines.get(key, {})
            if baseline and conc:
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
        out_path.write_text(json.dumps(result, indent=2))
        logger.info(f"[Concurrent] Saved → {out_path.name}")
        logger.info(f"[Concurrent] Degradation: {json.dumps(degradation, indent=2)}")
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
        metrics = {"type": t_type, "model": model}

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
        script = f"""
import torch, time, json
from diffusers import AutoPipelineForText2Image

pipe = AutoPipelineForText2Image.from_pretrained(
    \"{model}\", torch_dtype=torch.float16, variant=\"fp16\"
).to(\"cuda\")

t_end = time.time() + {duration_s}
n_images = 0
total_steps = 0
steps = 10

while time.time() < t_end:
    pipe(prompt=\"a red apple\", num_inference_steps=steps)
    n_images += 1
    total_steps += steps

print(json.dumps({{
    \"n_images\": n_images,
    \"total_steps\": total_steps,
    \"it_per_s\": round(total_steps / {duration_s}, 3),
}}))
"""
        p = self.output_dir / "_diff_concurrent.py"
        p.write_text(script)
        try:
            out = subprocess.check_output(
                [sys.executable, str(p)],
                timeout=duration_s + 30,
                **_subprocess_kwargs(),
            )
            return json.loads(out.decode().strip().splitlines()[-1])
        except Exception:
            return {}

    def _run_baselines(self) -> dict:
        logger.info("[Concurrent] Collecting single-task baselines ...")
        baselines = {}
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
    """一键运行所有 Phase，输出统一的汇总报告。"""

    def __init__(
        self,
        output_dir: str = "results",
        gpu_name: str = "GPU",
        gpu_index: int = 0,
        skip_phases: list[str] | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_name = gpu_name
        self.gpu_index = gpu_index
        self.skip_phases = skip_phases or []
        self.all_results = []
        configure_logging(self.output_dir)
        self.config = config_manager.load_config()

    def run_all(self):
        logger.info("=" * 60)
        logger.info("  GPU Benchmark Suite")
        logger.info(f"  Target : {self.gpu_name}")
        logger.info(f"  Output : {self.output_dir}")
        logger.info(f"  Start  : {datetime.now().isoformat(timespec='seconds')}")
        logger.info("=" * 60)

        from .llm import LLMBenchmark, LLMContextScaleBenchmark
        from .multimodal import DiffusionBenchmark, CVBenchmark, ASRBenchmark

        common = dict(
            output_dir=str(self.output_dir),
            gpu_index=self.gpu_index,
            sample_interval_s=self.config.sample_interval_s,
            warmup_s=self.config.warmup_s,
        )

        phases = {
            "llm": lambda: LLMBenchmark(
                model_name=self.config.llm_model,
                precision=self.config.llm_precision,
                prompts=self.config.llm_prompts,
                **common,
            ).run(),
            "llm_fp16": lambda: LLMBenchmark(
                model_name=self.config.llm_model,
                precision=self.config.llm_fp16_precision,
                prompts=self.config.llm_prompts,
                **common,
            ).run(),
            "llm_context_scale": lambda: LLMContextScaleBenchmark(
                model_name=self.config.llm_model,
                precision=self.config.llm_precision,
                **common,
            ).run(),
            "diffusion": lambda: DiffusionBenchmark(
                model_name=self.config.diffusion_model,
                precision=self.config.diffusion_precision,
                n_images=self.config.diffusion_n_images,
                n_steps=self.config.diffusion_n_steps,
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
        json_data = json.dumps(report, indent=2, default=lambda o: o.__dict__)
        report_path.write_text(json_data, encoding="utf-8")
        logger.info(f"[Suite] Final report → {report_path}")
        logger.info(f"[Suite] All done. Results in: {self.output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPU AI Benchmark Suite")
    parser.add_argument("--gpu-name", default="GPU", help="Friendly name for the GPU")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--skip", nargs="*", default=[],
        help="Phase names to skip, e.g. --skip diffusion asr",
    )
    parser.add_argument(
        "--only", nargs="*", default=None,
        help="Run only these phases, e.g. --only llm cv",
    )
    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    skip = args.skip
    if args.only:
        all_phases = ["llm", "llm_fp16", "llm_context_scale", "diffusion", "cv", "asr", "concurrent"]
        skip = [p for p in all_phases if p not in args.only]

    suite = FullBenchmarkSuite(
        output_dir=args.output_dir,
        gpu_name=args.gpu_name,
        gpu_index=args.gpu_index,
        skip_phases=skip,
    )
    suite.run_all()


if __name__ == "__main__":
    main()
