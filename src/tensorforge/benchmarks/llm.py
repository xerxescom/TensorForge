from pathlib import Path
import sys

<<<<<<<< HEAD:src/tensorforge/benchmarks/llm.py
<<<<<<<< HEAD:src/tensorforge/benchmarks/llm.py
from ..core.tf_logger import logger

from ..core.collector import BenchmarkRunner, _subprocess_kwargs

IS_WINDOWS = sys.platform == "win32"


class LLMBenchmark(BenchmarkRunner):
    """
    用 ollama CLI 跑本地 LLM，自动采集:
      - tokens/s (生成速度)
      - TTFT (Time To First Token)
      - total_tokens_generated
      - tokens_per_joule (由基类自动派生)

    使用示例:
        bench = LLMBenchmark(
            model_name="llama3.1:8b",
            precision="q4_k_m",
            prompts=["用中文写一篇200字的科技短文"],
            n_runs=5,
        )
        result = bench.run()
    """

    DEFAULT_PROMPTS = [
        "Explain the difference between a transformer and an RNN in detail.",
        "Write a Python function to compute Fibonacci numbers recursively with memoization.",
        "Describe the water cycle in 300 words.",
        "What are the main causes and consequences of the French Revolution?",
        "Explain how gradient descent works in machine learning.",
    ]

    def __init__(
            self,
            model_name: str = "llama3.1:8b",
            precision: str = "q4_k_m",
            prompts: list[str] | None = None,
            n_runs: int = 5,
            context_lengths: list[int] | None = None,
            use_ollama: bool = True,
            local_model_path: str = None,
            **kwargs,
    ):
        super().__init__(
            task_name="llm_inference",
            model_name=model_name,
            precision=precision,
            **kwargs,
        )
        self.prompts = prompts or self.DEFAULT_PROMPTS
        self.n_runs = n_runs
        # 可选：测试不同上下文长度下的性能衰减
        self.context_lengths = context_lengths or []
        self.use_ollama = use_ollama
        self.local_model_path = local_model_path

        # 预检查模型可用性
        self._check_model_availability()

    def _check_model_availability(self):
        """检查模型可用性"""
        if self.use_ollama:
            # 检查 ollama 是否可用
            try:
                result = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    encoding="utf-8",
                    **_subprocess_kwargs(),
                )
                if result.returncode == 0:
                    logger.info(f"[llm] Ollama available, checking for model: {self.model_name}")
                    if self.model_name in result.stdout:
                        logger.info(f"[llm] Model {self.model_name} found in ollama")
                    else:
                        logger.warning(
                            f"[llm] Model {self.model_name} not found, will attempt to pull"
                        )
                else:
                    logger.warning("[llm] Ollama not available, consider using alternative backend")
            except Exception as e:
                logger.warning(f"[llm] Ollama check failed: {e}")
        else:
            # 检查本地模型文件
            if self.local_model_path and Path(self.local_model_path).exists():
                logger.info(f"[llm] Using local model: {self.local_model_path}")
            else:
                logger.warning("[llm] Local model not found, will try to download")

    def run_task(self) -> dict:
        logger.info(f"[llm] Starting LLM benchmark: {self.n_runs} runs with model {self.model_name}")
        start_time = time.time()
        run_results = []

        for i, prompt in enumerate(self.prompts[:self.n_runs]):
            logger.info(f"  LLM run {i + 1}/{self.n_runs} ...")
            run_start = time.time()
            r = self._single_run(prompt)
            run_elapsed = time.time() - run_start
            
            if r.get("error"):
                logger.warning(f"  LLM run {i + 1} failed: {r.get('error')}")
            else:
                logger.debug(f"  LLM run {i + 1} completed in {run_elapsed:.2f}s: {r.get('tokens_generated', 0)} tokens, {r.get('tokens_per_s', 0):.1f} tokens/s")
            
            run_results.append(r)

        total_elapsed = time.time() - start_time
        logger.info(f"[llm] All runs completed in {total_elapsed:.2f}s")
        
        # 聚合
        ttfts = [r["ttft_s"] for r in run_results if r["ttft_s"] > 0]
        tps_list = [r["tokens_per_s"] for r in run_results if r["tokens_per_s"] > 0]
        elapsed_list = [r["total_elapsed_s"] for r in run_results if r["total_elapsed_s"] > 0]
        total_toks = sum(r["tokens_generated"] for r in run_results)
        success_count = sum(1 for r in run_results if not r.get("error"))
        
        logger.info(f"[llm] Results: {success_count}/{len(run_results)} successful, {total_toks} total tokens")
        if tps_list:
            logger.info(f"[llm] Performance: {sum(tps_list)/len(tps_list):.1f} avg tokens/s, {max(tps_list):.1f} max tokens/s")

        def _percentile(values: list[float], q: float) -> float:
            if not values:
                return 0
            s = sorted(values)
            idx = min(max(int(len(s) * q), 0), len(s) - 1)
            return round(s[idx], 4 if q < 1 else 2)

        return {
            "success_count": success_count,
            "failure_count": len(run_results) - success_count,
            "success_rate": round(success_count / len(run_results), 4) if run_results else 0,
            "tokens_per_s_mean": round(sum(tps_list) / len(tps_list), 2) if tps_list else 0,
            "tokens_per_s_max": round(max(tps_list), 2) if tps_list else 0,
            "tokens_per_s_min": round(min(tps_list), 2) if tps_list else 0,
            "tokens_per_s_p50": _percentile(tps_list, 0.50),
            "tokens_per_s_p95": _percentile(tps_list, 0.95),
            "ttft_s_mean": round(sum(ttfts) / len(ttfts), 4) if ttfts else 0,
            "ttft_s_min": round(min(ttfts), 4) if ttfts else 0,
            "ttft_s_p95": _percentile(ttfts, 0.95),
            "response_elapsed_s_mean": round(sum(elapsed_list) / len(elapsed_list), 4) if elapsed_list else 0,
            "tokens_generated": total_toks,
            "n_runs": len(run_results),
            "per_run_detail": run_results,
        }

    def _single_run(self, prompt: str) -> dict:
        """
        调用 ollama run，解析速度统计。
        跨平台注意点：
          - Windows 不支持逐行读 Popen.stdout（会阻塞），改用 communicate() 一次性读取
          - ollama --verbose 在旧版 Windows 安装包中可能不支持，fallback 到计时估算
        """
        logger.debug(f"[llm] Starting single run with prompt length: {len(prompt)} chars")
        # ollama run 的参数在新版本统一，--verbose 输出 eval rate
        cmd = ["ollama", "run", self.model_name, prompt, "--verbose"]

        ttft_s = 0.0
        tokens_per_s = 0.0
        tokens_generated = 0
        t_start = time.perf_counter()
        total_elapsed_s = 0.0
        output_text = ""
        error = None
        tokens_estimated = False

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",  # <--- 明确指定使用 UTF-8
                bufsize=1,
                **_subprocess_kwargs(),
            )
            first_chunk = ""
            while True:
                ch = proc.stdout.read(1) if proc.stdout else ""
                if ch:
                    first_chunk = ch
                    ttft_s = round(time.perf_counter() - t_start, 4)
                    break
                if proc.poll() is not None:
                    break

            try:
                remaining_out, remaining_err = proc.communicate(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
                remaining_out, remaining_err = proc.communicate()
                raise

            output_text = first_chunk + (remaining_out or "")
            total_elapsed_s = round(time.perf_counter() - t_start, 4)

            if ttft_s == 0 and output_text:
                ttft_s = total_elapsed_s

            # 解析 ollama verbose stderr 统计行
            for line in (remaining_err or "").splitlines():
                line = line.strip()
                if "eval rate:" in line:
                    try:
                        tokens_per_s = float(
                            line.split("eval rate:")[1].split("tokens/s")[0].strip()
                        )
                    except Exception:
                        pass
                if "eval count:" in line:
                    try:
                        tokens_generated = int(line.split("eval count:")[1].split()[0])
                    except Exception:
                        pass

            # 如果 --verbose 没有输出 eval rate（旧版 ollama），用输出字数粗估
            if tokens_per_s == 0 and output_text:
                decode_elapsed = max(total_elapsed_s - ttft_s, 1e-6)
                est_tokens = int(len(output_text.split()) * 1.3)
                tokens_per_s = round(est_tokens / decode_elapsed, 2) if decode_elapsed > 0 else 0
                tokens_generated = tokens_generated or est_tokens
                tokens_estimated = True

        except FileNotFoundError:
            error = "ollama_not_found"
            logger.warning("  [warn] ollama not found. Install from https://ollama.com")
        except subprocess.TimeoutExpired:
            error = "ollama_timeout"
            total_elapsed_s = round(time.perf_counter() - t_start, 4)
            logger.warning("  [warn] ollama run timed out (120s)")
        except Exception as e:
            error = str(e)
            total_elapsed_s = round(time.perf_counter() - t_start, 4)
            logger.warning(f"  [warn] ollama error: {e}")

        return {
            "prompt_preview": prompt[:60] + "...",
            "ttft_s": ttft_s,
            "total_elapsed_s": total_elapsed_s,
            "tokens_per_s": tokens_per_s,
            "tokens_generated": tokens_generated,
            "tokens_estimated": tokens_estimated,
            "error": error,
        }


class LLMContextScaleBenchmark(BenchmarkRunner):
    """
    测试上下文长度从 512 → 32k 的性能衰减曲线
    记录每个长度下的 tokens/s
    """

    def __init__(
            self,
            model_name: str = "llama3.1:8b",
            precision: str = "q4_k_m",
            context_lengths: list[int] | None = None,
            **kwargs,
    ):
        super().__init__(
            task_name="llm_context_scale",
            model_name=model_name,
            precision=precision,
            **kwargs,
        )
        self.context_lengths = context_lengths or [512, 1024, 2048, 4096, 8192, 16384]

    def run_task(self) -> dict:
        scale_results = []
        base_word = "The quick brown fox jumps over the lazy dog. "

        for ctx_len in self.context_lengths:
            # 构造约 ctx_len token 的 prompt（粗略估算：1 word ≈ 1.3 tokens）
            n_words = int(ctx_len / 1.3)
            prompt = (base_word * (n_words // len(base_word.split()) + 1))
            prompt = " ".join(prompt.split()[:n_words])
            full_prompt = (
                "Read the following context carefully and reply with exactly one word: OK.\n\n"
                f"{prompt}"
            )

            logger.info(f"  Context scale test: {ctx_len} tokens ...")
            r = self._single_probe(full_prompt)

            scale_results.append({
                "context_tokens": ctx_len,
                "prefill_latency_s": r["ttft_s"],
                "total_elapsed_s": r["total_elapsed_s"],
                "output_tokens": r["tokens_generated"],
                "decode_tokens_per_s": r["tokens_per_s"],
                "error": r["error"],
            })

        return {
            "context_scale_curve": scale_results,
        }

    def _single_probe(self, prompt: str) -> dict:
        bench = LLMBenchmark(
            model_name=self.model_name,
            precision=self.precision,
            prompts=[prompt],
            n_runs=1,
            output_dir=str(self.output_dir),
            gpu_index=self.gpu_index,
            sample_interval_s=self.sample_interval_s,
            warmup_s=0,
            keep_raw_samples=False,
        )
        return bench._single_run(prompt)
========
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensorforge.benchmarks.llm import *
>>>>>>>> 48d5659 (Refactor project into package structure):llm_bench.py
========
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensorforge.benchmarks.llm import *
>>>>>>>> origin/Aivor:llm_bench.py
