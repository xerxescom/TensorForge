"""
LLM inference speed benchmark.
"""

from __future__ import annotations

import contextlib
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from .collector import BenchmarkRunner, _subprocess_kwargs
from .error_schema import classify_error_type, short_trace
from .tf_logger import logger
from .token_counting import heuristic_token_count, local_exact_token_count


class LLMBenchmark(BenchmarkRunner):
    DEFAULT_PROMPTS: ClassVar[list[str]] = [
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
        local_model_path: str | None = None,
        ttft_sampling_mode: str = "chunk",
        **kwargs,
    ):
        super().__init__(
            task_name="llm_inference", model_name=model_name, precision=precision, **kwargs
        )
        self.prompts = prompts or self.DEFAULT_PROMPTS
        self.n_runs = n_runs
        self.context_lengths = context_lengths or []
        self.use_ollama = use_ollama
        self.local_model_path = local_model_path
        self.ttft_sampling_mode = ttft_sampling_mode.lower()
        self._check_model_availability()

    def _check_model_availability(self):
        if self.use_ollama:
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
                    if self.model_name not in result.stdout:
                        logger.warning(
                            f"[llm] Model {self.model_name} not found in ollama, may need pull"
                        )
                else:
                    logger.warning("[llm] Ollama not available, consider alternative backend")
            except Exception as e:
                logger.warning(f"[llm] Ollama check failed: {e}")
        else:
            if self.local_model_path and Path(self.local_model_path).exists():
                logger.info(f"[llm] Using local model: {self.local_model_path}")
            else:
                logger.warning("[llm] Local model not found, will try to download")

    def run_task(self) -> dict:
        start_time = time.time()
        run_results = []

        for i, prompt in enumerate(self.prompts[: self.n_runs]):
            logger.info(f"  LLM run {i + 1}/{self.n_runs} ...")
            r = self._single_run(prompt)
            run_results.append(r)

        ttfts = [r["ttft_s"] for r in run_results if r["ttft_s"] > 0]
        tps_list = [r["tokens_per_s"] for r in run_results if r["tokens_per_s"] > 0]
        elapsed_list = [r["total_elapsed_s"] for r in run_results if r["total_elapsed_s"] > 0]
        total_toks = sum(r["tokens_generated"] for r in run_results)
        total_toks_exact = sum(r.get("tokens_generated_exact", 0) for r in run_results)
        total_toks_estimated = sum(r.get("tokens_generated_estimated", 0) for r in run_results)
        success_count = sum(1 for r in run_results if not r.get("error"))
        exact_count = sum(1 for r in run_results if r.get("token_count_precision") == "exact")
        estimated_count = len(run_results) - exact_count

        def _percentile(values: list[float], q: float) -> float:
            if not values:
                return 0
            s = sorted(values)
            idx = min(max(int(len(s) * q), 0), len(s) - 1)
            return round(s[idx], 4 if q < 1 else 2)

        _ = time.time() - start_time
        return {
            "success_count": success_count,
            "failure_count": len(run_results) - success_count,
            "success_rate": round(success_count / len(run_results), 4) if run_results else 0,
            "ttft_sampling_mode": self.ttft_sampling_mode,
            "tokens_per_s_mean": round(sum(tps_list) / len(tps_list), 2) if tps_list else 0,
            "tokens_per_s_max": round(max(tps_list), 2) if tps_list else 0,
            "tokens_per_s_min": round(min(tps_list), 2) if tps_list else 0,
            "tokens_per_s_p50": _percentile(tps_list, 0.50),
            "tokens_per_s_p95": _percentile(tps_list, 0.95),
            "ttft_s_mean": round(sum(ttfts) / len(ttfts), 4) if ttfts else 0,
            "ttft_s_min": round(min(ttfts), 4) if ttfts else 0,
            "ttft_s_p95": _percentile(ttfts, 0.95),
            "response_elapsed_s_mean": round(sum(elapsed_list) / len(elapsed_list), 4)
            if elapsed_list
            else 0,
            "tokens_generated": total_toks,
            "tokens_generated_exact": total_toks_exact,
            "tokens_generated_estimated": total_toks_estimated,
            "token_count_precision_breakdown": {
                "exact": exact_count,
                "estimated": estimated_count,
            },
            "n_runs": len(run_results),
            "per_run_detail": run_results,
        }

    def _single_run(self, prompt: str) -> dict:
        cmd = ["ollama", "run", self.model_name, prompt, "--verbose"]

        ttft_s = 0.0
        tokens_per_s = 0.0
        tokens_generated = 0
        t_start = time.perf_counter()
        total_elapsed_s = 0.0
        output_text = ""
        error = None
        tokens_estimated = False
        token_count_precision = "exact"
        token_count_source = "backend_usage"
        tokens_generated_exact = 0
        tokens_generated_estimated = 0
        prompt_tokens = 0

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
                **_subprocess_kwargs(),
            )
            output_chunks, first_output_ts, sampling_mode = self._sample_stdout(
                proc, on_data=lambda _data: None
            )
            self.ttft_sampling_mode = sampling_mode

            remaining_err = proc.stderr.read() if proc.stderr else ""
            output_text = "".join(output_chunks)
            total_elapsed_s = round(time.perf_counter() - t_start, 4)
            if first_output_ts is not None:
                ttft_s = round(first_output_ts - t_start, 4)

            if ttft_s == 0 and output_text:
                ttft_s = total_elapsed_s

            parsed_usage = self._parse_ollama_verbose_usage(remaining_err or "")
            prompt_tokens = parsed_usage["prompt_tokens"]
            tokens_generated_exact = parsed_usage["completion_tokens"]
            tokens_generated = tokens_generated_exact
            if parsed_usage["eval_rate"] > 0:
                tokens_per_s = parsed_usage["eval_rate"]

            if tokens_generated == 0 and output_text:
                local_tokens, source = local_exact_token_count(
                    output_text,
                    model_name=self.model_name,
                    local_model_path=self.local_model_path,
                )
                if local_tokens is not None:
                    tokens_generated_exact = local_tokens
                    tokens_generated = local_tokens
                    token_count_source = source or "local_tokenizer"
                else:
                    tokens_generated_estimated = heuristic_token_count(output_text)
                    tokens_generated = tokens_generated_estimated
                    tokens_estimated = True
                    token_count_precision = "estimated"
                    token_count_source = "heuristic"

            if tokens_per_s == 0 and output_text and tokens_generated > 0:
                decode_elapsed = max(total_elapsed_s - ttft_s, 1e-6)
                tokens_per_s = (
                    round(tokens_generated / decode_elapsed, 2) if decode_elapsed > 0 else 0
                )

            if tokens_generated > 0 and not tokens_estimated and tokens_generated_exact == 0:
                tokens_generated_exact = tokens_generated

        except FileNotFoundError:
            error = "ollama_not_found"
        except subprocess.TimeoutExpired:
            error = "ollama_timeout"
            total_elapsed_s = round(time.perf_counter() - t_start, 4)
        except Exception as e:
            error = str(e)
            total_elapsed_s = round(time.perf_counter() - t_start, 4)

        return {
            "prompt_preview": prompt[:60] + "...",
            "ttft_s": ttft_s,
            "ttft_sampling_mode": self.ttft_sampling_mode,
            "total_elapsed_s": total_elapsed_s,
            "tokens_per_s": tokens_per_s,
            "tokens_generated": tokens_generated,
            "tokens_generated_exact": tokens_generated_exact,
            "tokens_generated_estimated": tokens_generated_estimated,
            "prompt_tokens": prompt_tokens,
            "token_count_precision": token_count_precision,
            "token_count_source": token_count_source,
            "tokens_estimated": tokens_estimated,
            "error": error,
            "error_type": classify_error_type(str(error)) if error else None,
            "error_stage": "llm_subprocess" if error else None,
            "short_trace": short_trace(str(error)) if error else None,
        }

    def _parse_ollama_verbose_usage(self, stderr_text: str) -> dict[str, float | int]:
        prompt_tokens = 0
        completion_tokens = 0
        eval_rate = 0.0
        for line in stderr_text.splitlines():
            line = line.strip()
            if "prompt eval count:" in line:
                with contextlib.suppress(Exception):
                    prompt_tokens = int(line.split("prompt eval count:")[1].split()[0])
            if "eval count:" in line and "prompt eval count:" not in line:
                with contextlib.suppress(Exception):
                    completion_tokens = int(line.split("eval count:")[1].split()[0])
            if "eval rate:" in line:
                with contextlib.suppress(Exception):
                    eval_rate = float(line.split("eval rate:")[1].split("tokens/s")[0].strip())
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "eval_rate": eval_rate,
        }

    def _resolve_sampling_mode(self) -> str:
        mode = self.ttft_sampling_mode
        return mode if mode in {"char", "chunk", "line"} else "chunk"

    def _sample_stdout(
        self, proc: subprocess.Popen, on_data: Callable[[str], None]
    ) -> tuple[list[str], float | None, str]:
        """
        Sample stdout with configured mode and record first non-empty output timestamp.
        """
        output_chunks: list[str] = []
        first_output_ts: float | None = None
        mode = self._resolve_sampling_mode()

        def _consume(data: str):
            nonlocal first_output_ts
            if not data:
                return
            if first_output_ts is None and data.strip():
                first_output_ts = time.perf_counter()
            output_chunks.append(data)
            on_data(data)

        def _reader():
            if not proc.stdout:
                return
            read_fn = {
                "char": lambda: proc.stdout.read(1),
                "line": proc.stdout.readline,
                "chunk": lambda: proc.stdout.read(256),
            }[mode]
            while True:
                buf = read_fn()
                if not buf:
                    break
                _consume(buf)

        reader = threading.Thread(target=_reader, daemon=True)
        reader.start()
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise
        finally:
            reader.join(timeout=2)
        return output_chunks, first_output_ts, mode


class LLMContextScaleBenchmark(BenchmarkRunner):
    def __init__(
        self,
        model_name: str = "llama3.1:8b",
        precision: str = "q4_k_m",
        context_lengths: list[int] | None = None,
        **kwargs,
    ):
        super().__init__(
            task_name="llm_context_scale", model_name=model_name, precision=precision, **kwargs
        )
        self.context_lengths = context_lengths or [512, 1024, 2048, 4096, 8192, 16384]

    def run_task(self) -> dict:
        scale_results = []
        base_word = "The quick brown fox jumps over the lazy dog. "

        for ctx_len in self.context_lengths:
            n_words = int(ctx_len / 1.3)
            prompt = base_word * (n_words // len(base_word.split()) + 1)
            prompt = " ".join(prompt.split()[:n_words])
            full_prompt = (
                "Read the following context carefully and reply with exactly one word: OK.\n\n"
                f"{prompt}"
            )
            logger.info(f"  Context scale test: {ctx_len} tokens ...")
            r = self._single_probe(full_prompt)
            scale_results.append(
                {
                    "context_tokens": ctx_len,
                    "prefill_latency_s": r["ttft_s"],
                    "total_elapsed_s": r["total_elapsed_s"],
                    "output_tokens": r["tokens_generated"],
                    "decode_tokens_per_s": r["tokens_per_s"],
                    "error": r["error"],
                }
            )
        return {"context_scale_curve": scale_results}

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
