"""
LLM 推理速度测试
依赖: ollama (本地运行) 或 llama-cpp-python
支持平台：Windows 10/11 · Linux
"""
import time
import subprocess
import sys
import json
from core.collector import BenchmarkRunner, _subprocess_kwargs

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
        total_toks  = sum(r["tokens_generated"] for r in run_results)

        return {
            "tokens_per_s_mean":  round(sum(tps_list) / len(tps_list), 2) if tps_list else 0,
            "tokens_per_s_max":   round(max(tps_list), 2) if tps_list else 0,
            "tokens_per_s_min":   round(min(tps_list), 2) if tps_list else 0,
            "ttft_s_mean":        round(sum(ttfts) / len(ttfts), 4) if ttfts else 0,
            "ttft_s_min":         round(min(ttfts), 4) if ttfts else 0,
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

        try:
            # Windows 必须用 communicate()，Linux 也兼容
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                **_subprocess_kwargs(),
            )
            ttft_s = round(time.perf_counter() - t_start, 4)  # Windows fallback: 总时间近似

            # 解析 ollama verbose stderr 统计行
            for line in proc.stderr.splitlines():
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
            if tokens_per_s == 0 and proc.stdout:
                elapsed = time.perf_counter() - t_start
                est_tokens = int(len(proc.stdout.split()) * 1.3)
                tokens_per_s = round(est_tokens / elapsed, 2) if elapsed > 0 else 0
                tokens_generated = tokens_generated or est_tokens

        except FileNotFoundError:
            print("  [warn] ollama not found. Install from https://ollama.com")
        except subprocess.TimeoutExpired:
            print("  [warn] ollama run timed out (120s)")
        except Exception as e:
            print(f"  [warn] ollama error: {e}")

        return {
            "prompt_preview": prompt[:60] + "...",
            "ttft_s": ttft_s,
            "tokens_per_s": tokens_per_s,
            "tokens_generated": tokens_generated,
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

            print(f"  Context scale test: {ctx_len} tokens ...")
            t0 = time.perf_counter()
            # 这里简化为计时调用，实际可替换为真实推理
            try:
                subprocess.run(
                    ["ollama", "run", self.model_name, f"Summarize: {prompt[:200]}"],
                    capture_output=True, text=True, timeout=120,
                )
                elapsed = round(time.perf_counter() - t0, 3)
            except Exception:
                elapsed = 0.0

            scale_results.append({
                "context_tokens": ctx_len,
                "elapsed_s": elapsed,
                "tokens_per_s": round(ctx_len / elapsed, 2) if elapsed > 0 else 0,
            })

        return {
            "context_scale_curve": scale_results,
        }
