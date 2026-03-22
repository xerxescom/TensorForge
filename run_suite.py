"""
多任务并发压测 + 完整测试套件入口
支持平台：Windows 10/11 · Linux
"""
import time
import json
import sys
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from collector import GPUSampler, _subprocess_kwargs


# ─────────────────────────────────────────────────────────────
#  并发压测
# ─────────────────────────────────────────────────────────────
class ConcurrentStressTest:
    """
    同时运行多个推理任务，测量:
      - 各任务吞吐量在并发时的衰减率
      - VRAM 峰值与 OOM 边界
      - 功耗在混合负载下的行为

    使用示例:
        test = ConcurrentStressTest(
            tasks=[
                {"type": "llm",      "model": "llama3.1:8b", "duration_s": 60},
                {"type": "diffusion","model": "sdxl-turbo",   "duration_s": 60},
            ],
            output_dir="results",
        )
        result = test.run()
    """

    def __init__(self, tasks: list[dict], output_dir: str = "results", gpu_index: int = 0):
        self.tasks = tasks
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_index = gpu_index

    def run(self) -> dict:
        print(f"\n[Concurrent] Starting {len(self.tasks)} tasks simultaneously ...")
        sampler = GPUSampler(interval_s=0.5, gpu_index=self.gpu_index)

        # 先跑基准（单任务）
        baselines = self._run_baselines()

        # 并发跑
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

        # 计算衰减率
        degradation = {}
        for i, (cfg, conc) in enumerate(zip(self.tasks, concurrent_metrics)):
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

        # 保存
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.output_dir / f"concurrent_{ts}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"[Concurrent] Saved → {out_path.name}")
        print(f"[Concurrent] Degradation: {json.dumps(degradation, indent=2)}")
        return result

    @staticmethod
    def _safe_ratio(num, den):
        if num is None or den is None or den <= 0:
            return None
        return round(num / den, 4)

    def _run_one_task(self, cfg: dict, idx: int, results: list, barrier: threading.Barrier):
        """在独立线程中跑单个任务"""
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
        """在 duration_s 内循环跑 LLM，统计总 tokens"""
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
                    **_subprocess_kwargs(),
                )
                total_tokens += int(len(out.stdout.split()) * 1.3)
            except Exception:
                break

        return {
            "tokens_generated": total_tokens,
            "tokens_per_s": round(total_tokens / duration_s, 2),
        }

    def _timed_diffusion(self, model: str, duration_s: float) -> dict:
        """在 duration_s 内循环生图，统计 iterations"""
        script = f"""
import torch, time, json
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
        """依次单独跑，获取基准性能"""
        print("[Concurrent] Collecting single-task baselines ...")
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


# ─────────────────────────────────────────────────────────────
#  完整测试套件入口
# ─────────────────────────────────────────────────────────────
class FullBenchmarkSuite:
    """
    一键运行所有 Phase，输出统一的汇总报告

    使用示例:
        suite = FullBenchmarkSuite(
            output_dir="results/rtx4090_20250319",
            gpu_name="RTX 4090",
            gpu_index=0,
        )
        suite.run_all()
    """

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

    def run_all(self):
        print(f"\n{'='*60}")
        print(f"  GPU Benchmark Suite")
        print(f"  Target : {self.gpu_name}")
        print(f"  Output : {self.output_dir}")
        print(f"  Start  : {datetime.now().isoformat(timespec='seconds')}")
        print(f"{'='*60}\n")

        from llm_bench import LLMBenchmark, LLMContextScaleBenchmark
        from other_bench import DiffusionBenchmark, CVBenchmark, ASRBenchmark

        common = dict(output_dir=str(self.output_dir), gpu_index=self.gpu_index)

        phases = {
            "llm": lambda: LLMBenchmark(
                model_name="llama3.1:8b", precision="q4_k_m",
                n_runs=5, warmup_s=5, **common,
            ).run(),

            "llm_fp16": lambda: LLMBenchmark(
                model_name="llama3.1:8b", precision="fp16",
                n_runs=5, warmup_s=5, **common,
            ).run(),

            "llm_context_scale": lambda: LLMContextScaleBenchmark(
                model_name="llama3.1:8b", warmup_s=3, **common,
            ).run(),

            "diffusion": lambda: DiffusionBenchmark(
                model_name="stabilityai/sdxl-turbo", precision="fp16",
                n_images=10, n_steps=20, warmup_s=10, **common,
            ).run(),

            "cv": lambda: CVBenchmark(
                model_name="yolov8n", precision="fp16",
                n_frames=200, warmup_s=5, **common,
            ).run(),

            "asr": lambda: ASRBenchmark(
                model_name="base", precision="float16",
                warmup_s=5, **common,
            ).run(),

            "concurrent": lambda: ConcurrentStressTest(
                tasks=[
                    {"type": "llm",      "model": "llama3.1:8b",         "duration_s": 60},
                    {"type": "diffusion","model": "stabilityai/sdxl-turbo","duration_s": 60},
                ],
                output_dir=str(self.output_dir),
                gpu_index=self.gpu_index,
            ).run(),
        }

        for phase_name, phase_fn in phases.items():
            if phase_name in self.skip_phases:
                print(f"[Suite] Skipping phase: {phase_name}")
                continue
            print(f"\n[Suite] ── Phase: {phase_name} ──")
            try:
                result = phase_fn()
                self.all_results.append({"phase": phase_name, "result": result})
            except Exception as e:
                print(f"[Suite] ERROR in {phase_name}: {e}")
                self.all_results.append({"phase": phase_name, "error": str(e)})

        self._write_final_report()

    def _write_final_report(self):
        report = {
            "gpu_name": self.gpu_name,
            "benchmark_date": datetime.now().isoformat(timespec="seconds"),
            "phases": self.all_results,
        }
        report_path = self.output_dir / "final_report.json"
        # 告诉 json 模块：遇到不认识的对象，就直接调用它的 __dict__ 属性转化为字典
        json_data = json.dumps(
            report,
            indent=2,
            default=lambda o: o.__dict__
        )

        # 写入文件，务必带上 utf-8 编码！
        report_path.write_text(json_data, encoding="utf-8")
        # report_path.write_text(json.dumps(report, indent=2))
        print(f"\n[Suite] Final report → {report_path}")
        print(f"[Suite] All done. Results in: {self.output_dir}")


# ─────────────────────────────────────────────────────────────
#  命令行入口
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPU AI Benchmark Suite")
    parser.add_argument("--gpu-name", default="GPU", help="Friendly name for the GPU")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--skip", nargs="*", default=[],
        help="Phase names to skip, e.g. --skip diffusion asr"
    )
    parser.add_argument(
        "--only", nargs="*", default=None,
        help="Run only these phases, e.g. --only llm cv"
    )
    args = parser.parse_args()

    skip = args.skip
    if args.only:
        all_phases = ["llm", "llm_fp16", "llm_context_scale",
                      "diffusion", "cv", "asr", "concurrent"]
        skip = [p for p in all_phases if p not in args.only]

    suite = FullBenchmarkSuite(
        output_dir=args.output_dir,
        gpu_name=args.gpu_name,
        gpu_index=args.gpu_index,
        skip_phases=skip,
    )
    suite.run_all()
