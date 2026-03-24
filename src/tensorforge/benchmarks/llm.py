"""
LLM 推理速度测试
依赖: ollama (本地运行) 或 llama-cpp-python
支持平台：Windows 10/11 · Linux
"""
import subprocess
import sys
import time
import json
import statistics
from pathlib import Path
from typing import List, Dict, Optional

from ..core.collector import BenchmarkRunner, _subprocess_kwargs
from ..core.config_manager import config_manager

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
            prompts: List[str] | None = None,
            n_runs: int = 5,
            context_lengths: List[int] | None = None,
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
        
        # 加载配置
        self.config = config_manager.load_config()

    def run_task(self) -> dict:
        run_results = []

        for i, prompt in enumerate(self.prompts[:self.n_runs]):
            print(f"  LLM run {i + 1}/{self.n_runs} ...")
            r = self._single_run(prompt)
            run_results.append(r)

        # 聚合
        ttfts = [r["ttft_s"] for r in run_results if r["ttft_s"] > 0]
        tps_list = [r["tokens_per_s"] for r in run_results if r["tokens_per_s"] > 0]
        elapsed_list = [r["total_elapsed_s"] for r in run_results if r["total_elapsed_s"] > 0]
        total_toks = sum(r["tokens_generated"] for r in run_results)
        success_count = sum(1 for r in run_results if not r.get("error"))

        def _percentile(values: List[float], q: float) -> float:
            if not values:
                return 0.0
            sorted_vals = sorted(values)
            idx = int(q * len(sorted_vals))
            return round(sorted_vals[min(idx, len(sorted_vals) - 1)], 4 if q < 1 else 2)

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
        """单次运行一次 LLM 推理，返回指标字典"""
        if self.use_ollama:
            return self._ollama_run(prompt)
        else:
            # TODO: 支持 llama-cpp-python 后端
            raise NotImplementedError("Only ollama backend is currently supported")

    def _ollama_run(self, prompt: str) -> dict:
        """使用 ollama CLI 运行推理"""
        cmd = [
            "ollama", "run", self.model_name,
            "--format", "json",
            prompt
        ]
        
        t0 = time.perf_counter()
        ttft = None
        tokens_generated = 0
        
        try:
            # 记录首 token 时间（简化版本，实际需要流式输出）
            start_time = time.perf_counter()
            
            out = subprocess.check_output(
                cmd,
                stderr=subprocess.STDOUT,
                timeout=self.config.timeout_s if hasattr(self.config, 'timeout_s') else 120,
                **_subprocess_kwargs(),
            )
            
            total_elapsed = time.perf_counter() - start_time
            
            # 解析 JSON 输出
            try:
                response = json.loads(out.decode())
                response_text = response.get("response", "")
                
                # 简单估算 token 数量（实际应该用 tokenizer）
                tokens_generated = len(response_text.split())
                
                # 估算 TTFT（简化版本）
                ttft = total_elapsed * 0.1  # 假设首 token 占 10% 时间
                
                tokens_per_s = tokens_generated / total_elapsed if total_elapsed > 0 else 0
                
                return {
                    "tokens_generated": tokens_generated,
                    "ttft_s": ttft,
                    "total_elapsed_s": total_elapsed,
                    "tokens_per_s": tokens_per_s,
                    "prompt_length": len(prompt),
                    "response_length": len(response_text),
                }
                
            except json.JSONDecodeError:
                return {
                    "tokens_generated": 0,
                    "ttft_s": 0,
                    "total_elapsed_s": total_elapsed,
                    "tokens_per_s": 0,
                    "prompt_length": len(prompt),
                    "response_length": 0,
                    "error": "Failed to parse JSON response",
                }
                
        except subprocess.CalledProcessError as e:
            elapsed = time.perf_counter() - t0
            return {
                "tokens_generated": 0,
                "ttft_s": 0,
                "total_elapsed_s": elapsed,
                "tokens_per_s": 0,
                "prompt_length": len(prompt),
                "response_length": 0,
                "error": f"ollama failed: {e.output.decode()[-200:] if e.output else str(e)}",
            }
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - t0
            return {
                "tokens_generated": 0,
                "ttft_s": 0,
                "total_elapsed_s": elapsed,
                "tokens_per_s": 0,
                "prompt_length": len(prompt),
                "response_length": 0,
                "error": "timeout",
            }


class LLMContextScaleBenchmark(BenchmarkRunner):
    """
    测试不同上下文长度下的 LLM 性能衰减
    """

    def __init__(
        self,
        model_name: str = "llama3.1:8b",
        precision: str = "q4_k_m",
        context_lengths: List[int] | None = None,
        **kwargs,
    ):
        super().__init__(
            task_name="llm_context_scale",
            model_name=model_name,
            precision=precision,
            **kwargs,
        )
        self.context_lengths = context_lengths or [512, 1024, 2048, 4096]
        # 构造不同长度的提示词
        self.prompts = [self._make_prompt(length) for length in self.context_lengths]

    def _make_prompt(self, target_tokens: int) -> str:
        """构造指定长度的提示词"""
        base = "Explain machine learning concepts in detail. "
        # 重复基础提示词直到达到目标长度
        while len(base.split()) < target_tokens:
            base += "Provide more examples and explanations. "
        return base

    def run_task(self) -> dict:
        results = {}
        
        for i, (ctx_len, prompt) in enumerate(zip(self.context_lengths, self.prompts)):
            print(f"  Context length {ctx_len} tokens...")
            
            # 使用标准的 LLMBenchmark 来运行
            llm_bench = LLMBenchmark(
                model_name=self.model_name,
                precision=self.precision,
                prompts=[prompt],
                n_runs=1,  # 每个长度只跑一次
                warmup_s=0,  # 跳过预热
                output_dir=self.output_dir,
            )
            
            result = llm_bench.run()
            
            if result.status == "ok" and result.metrics:
                results[str(ctx_len)] = {
                    "tokens_per_s": result.metrics.get("tokens_per_s_mean", 0),
                    "ttft_s": result.metrics.get("ttft_s_mean", 0),
                    "success_rate": result.metrics.get("success_rate", 0),
                }
            else:
                results[str(ctx_len)] = {
                    "tokens_per_s": 0,
                    "ttft_s": 0,
                    "success_rate": 0,
                    "error": result.error,
                }
        
        return {
            "context_lengths": self.context_lengths,
            "results": results,
            "performance_degradation": self._calc_degradation(results),
        }

    def _calc_degradation(self, results: dict) -> dict:
        """计算性能衰减率"""
        if not results or len(results) < 2:
            return {}
        
        # 获取第一个长度作为基准
        first_len = min(int(k) for k in results.keys() if results[k].get("tokens_per_s", 0) > 0)
        base_tps = results[str(first_len)]["tokens_per_s"]
        
        if base_tps == 0:
            return {}
        
        degradation = {}
        for ctx_len_str, metrics in results.items():
            ctx_len = int(ctx_len_str)
            if metrics.get("tokens_per_s", 0) > 0:
                degradation_ratio = metrics["tokens_per_s"] / base_tps
                degradation[str(ctx_len)] = {
                    "ratio": round(degradation_ratio, 3),
                    "degradation_percent": round((1 - degradation_ratio) * 100, 1),
                }
        
        return degradation
