"""
LLM 推理速度测试
依赖: ollama (本地运行) 或 llama-cpp-python
支持平台：Windows 10/11 · Linux
"""
import time
import subprocess
import sys
from collector import BenchmarkRunner, _subprocess_kwargs

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

    def run_task(self) -> dict:
        run_results = []

        for i, prompt in enumerate(self.prompts[:self.n_runs]):
            print(f"  LLM run {i+1}/{self.n_runs} ...")
            r = self._single_run(prompt)
            run_results.append(r)

        # 聚合
        ttfts       = [r["ttft_s"]       for r in run_results if r["ttft_s"] > 0]
        tps_list    = [r["tokens_per_s"]  for r in run_results if r["tokens_per_s"] > 0]
        elapsed_list = [r["total_elapsed_s"] for r in run_results if r["total_elapsed_s"] > 0]
        total_toks  = sum(r["tokens_generated"] for r in run_results)
        success_count = sum(1 for r in run_results if not r.get("error"))

        def _percentile(values: list[float], q: float) -> float:
            if not values:
                return 0
            s = sorted(values)
            idx = min(max(int(len(s) * q), 0), len(s) - 1)
            return round(s[idx], 4 if q < 1 else 2)

        return {
            "success_count":       success_count,
            "failure_count":       len(run_results) - success_count,
            "success_rate":        round(success_count / len(run_results), 4) if run_results else 0,
            "tokens_per_s_mean":  round(sum(tps_list) / len(tps_list), 2) if tps_list else 0,
            "tokens_per_s_max":   round(max(tps_list), 2) if tps_list else 0,
            "tokens_per_s_min":   round(min(tps_list), 2) if tps_list else 0,
            "tokens_per_s_p50":   _percentile(tps_list, 0.50),
            "tokens_per_s_p95":   _percentile(tps_list, 0.95),
            "ttft_s_mean":        round(sum(ttfts) / len(ttfts), 4) if ttfts else 0,
            "ttft_s_min":         round(min(ttfts), 4) if ttfts else 0,
            "ttft_s_p95":         _percentile(ttfts, 0.95),
            "response_elapsed_s_mean": round(sum(elapsed_list) / len(elapsed_list), 4) if elapsed_list else 0,
            "tokens_generated":   total_toks,
            "n_runs":             len(run_results),
            "per_run_detail":     run_results,
        }

    def _single_run(self, prompt: str) -> dict:
        """
        调用 ollama run，解析速度统计。
        跨平台注意点：
          - Windows 不支持逐行读 Popen.stdout（会阻塞），改用 communicate() 一次性读取
          - ollama --verbose 在旧版 Windows 安装包中可能不支持，fallback 到计时估算
        """
        # ollama run 的参数在新版本统一，--verbose 输出 eval rate
        cmd = ["ollama", "run", self.model_name, prompt, "--verbose"]

        ttft_s = 0.0
        tokens_per_s = 0.0
        tokens_generated = 0
        t_start = time.perf_counter()
        total_elapsed_s = 0.0
        output_text = ""
        error = None

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

        except FileNotFoundError:
            error = "ollama_not_found"
            print("  [warn] ollama not found. Install from https://ollama.com")
        except subprocess.TimeoutExpired:
            error = "ollama_timeout"
            total_elapsed_s = round(time.perf_counter() - t_start, 4)
            print("  [warn] ollama run timed out (120s)")
        except Exception as e:
            error = str(e)
            total_elapsed_s = round(time.perf_counter() - t_start, 4)
            print(f"  [warn] ollama error: {e}")

        return {
            "prompt_preview": prompt[:60] + "...",
            "ttft_s": ttft_s,
            "total_elapsed_s": total_elapsed_s,
            "tokens_per_s": tokens_per_s,
            "tokens_generated": tokens_generated,
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

            print(f"  Context scale test: {ctx_len} tokens ...")
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
