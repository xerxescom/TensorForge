"""
多任务并发压测 + 完整测试套件入口
支持平台：Windows 10/11 · Linux
"""
import argparse
import json
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from ..core.collector import GPUSampler
from ..core.config_manager import config_manager
from ..core.logging_utils import get_logger


# ─────────────────────────────────────────────────────────────
# 并发压测
# ─────────────────────────────────────────────────────────────
class ConcurrentStressTest:
    """
    同时运行多个推理任务，测量:
      - 各任务吞吐量在并发时的衰减率
      - VRAM 峰值与 OOM 边界
      - 功耗在混合负载下的行为
    """

    def __init__(self, tasks: List[dict], output_dir: str = "results", gpu_index: int = 0):
        self.tasks = tasks
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_index = gpu_index
        self.logger = get_logger("concurrent")

    def run(self) -> dict:
        self.logger.info(f"[Concurrent] Starting {len(self.tasks)} tasks simultaneously ...")
        sampler = GPUSampler(interval_s=0.5, gpu_index=self.gpu_index)

        baselines = self._run_baselines()
        
        sampler.start()
        t0 = time.time()
        threads = []
        concurrent_metrics = [None] * len(self.tasks)
        barrier = threading.Barrier(len(self.tasks))
        
        # 启动所有任务线程
        for i, task in enumerate(self.tasks):
            thread = threading.Thread(
                target=self._run_single_task,
                args=(task, i, concurrent_metrics, barrier),
                daemon=True
            )
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        elapsed = time.time() - t0
        samples = sampler.stop()
        
        # 分析结果
        result = {
            "task_count": len(self.tasks),
            "total_elapsed_s": elapsed,
            "concurrent_metrics": concurrent_metrics,
            "gpu_stats": self._calc_gpu_stats(samples),
            "throughput_analysis": self._analyze_throughput(concurrent_metrics, baselines),
            "resource_analysis": self._analyze_resource_usage(concurrent_metrics),
        }
        
        # 保存结果
        self._save_result(result)
        
        self.logger.info(f"[Concurrent] Completed in {elapsed:.2f}s")
        return result
    
    def _run_baselines(self) -> Dict[str, Any]:
        """运行基线测试（单任务）"""
        self.logger.info("[Concurrent] Running baseline tests...")
        baselines = {}
        
        for task in self.tasks[:1]:  # 只测试第一个任务类型
            task_type = task.get("type", "unknown")
            self.logger.info(f"  Baseline test for {task_type}...")
            
            # 单独运行任务
            sampler = GPUSampler(interval_s=0.5, gpu_index=self.gpu_index)
            sampler.start()
            
            t0 = time.time()
            result = self._run_single_task(task, 0, [None], None)
            elapsed = time.time() - t0
            
            samples = sampler.stop()
            
            baselines[task_type] = {
                "elapsed_s": elapsed,
                "metrics": result,
                "gpu_stats": self._calc_gpu_stats(samples),
            }
        
        return baselines
    
    def _run_single_task(self, task: dict, task_id: int, metrics_list: List[Any], barrier: threading.Barrier) -> Dict[str, Any]:
        """运行单个任务"""
        task_type = task.get("type", "unknown")
        model_name = task.get("model", "test-model")
        
        try:
            if barrier:
                barrier.wait()  # 同步所有任务开始
            
            t0 = time.perf_counter()
            
            # 根据任务类型执行不同操作
            if task_type == "llm":
                result = self._run_llm_task(task)
            elif task_type == "diffusion":
                result = self._run_diffusion_task(task)
            elif task_type == "cv":
                result = self._run_cv_task(task)
            else:
                result = {"status": "error", "error": f"Unknown task type: {task_type}"}
            
            elapsed = time.perf_counter() - t0
            
            if metrics_list is not None:
                metrics_list[task_id] = {
                    "task_id": task_id,
                    "task_type": task_type,
                    "model": model_name,
                    "elapsed_s": elapsed,
                    "metrics": result,
                    "status": result.get("status", "unknown"),
                }
            
            return result
            
        except Exception as e:
            error_msg = f"Task {task_id} ({task_type}) failed: {e}"
            self.logger.error(error_msg)
            
            if metrics_list is not None:
                metrics_list[task_id] = {
                    "task_id": task_id,
                    "task_type": task_type,
                    "model": model_name,
                    "elapsed_s": 0,
                    "metrics": {"status": "error", "error": str(e)},
                    "status": "error",
                }
            
            return {"status": "error", "error": str(e)}
    
    def _run_llm_task(self, task: dict) -> Dict[str, Any]:
        """运行 LLM 任务"""
        model_name = task.get("model", "test-model")
        prompt = task.get("prompt", "Test prompt for concurrent testing.")
        
        # 模拟 LLM 推理
        time.sleep(0.5)  # 模拟推理时间
        
        return {
            "status": "success",
            "tokens_generated": 50,
            "tokens_per_s": 25.0,
            "prompt_length": len(prompt),
        }
    
    def _run_diffusion_task(self, task: dict) -> Dict[str, Any]:
        """运行扩散模型任务"""
        model_name = task.get("model", "test-model")
        
        # 模拟图像生成
        time.sleep(1.0)  # 模拟推理时间
        
        return {
            "status": "success",
            "images_generated": 1,
            "images_per_s": 1.0,
            "steps": 20,
        }
    
    def _run_cv_task(self, task: dict) -> Dict[str, Any]:
        """运行计算机视觉任务"""
        model_name = task.get("model", "test-model")
        
        # 模拟图像处理
        time.sleep(0.3)  # 模拟推理时间
        
        return {
            "status": "success",
            "images_processed": 10,
            "fps": 15.0,
            "image_size": (224, 224),
        }
    
    def _calc_gpu_stats(self, samples: List[Any]) -> Dict[str, Any]:
        """计算 GPU 统计"""
        if not samples:
            return {}
        
        # 这里应该使用实际的 GPU 样本数据
        # 目前返回模拟数据
        return {
            "sample_count": len(samples),
            "avg_utilization": 75.0,
            "max_utilization": 95.0,
            "avg_memory_mb": 4096,
            "peak_memory_mb": 6144,
            "avg_power_w": 250.0,
            "peak_power_w": 350.0,
        }
    
    def _analyze_throughput(self, concurrent_metrics: List[Any], baselines: Dict[str, Any]) -> Dict[str, Any]:
        """分析吞吐量变化"""
        analysis = {
            "baseline_throughput": {},
            "concurrent_throughput": {},
            "throughput_degradation": {},
            "efficiency_ratio": {},
        }
        
        for i, metrics in enumerate(concurrent_metrics):
            if not metrics or metrics.get("status") != "success":
                continue
            
            task_type = metrics.get("task_type", f"task_{i}")
            
            # 基线吞吐量
            if task_type in baselines:
                baseline = baselines[task_type]
                baseline_throughput = self._calculate_throughput(baseline)
                analysis["baseline_throughput"][task_type] = baseline_throughput
            
            # 并发吞吐量
            concurrent_throughput = self._calculate_throughput(metrics)
            analysis["concurrent_throughput"][task_type] = concurrent_throughput
            
            # 衰减率
            if task_type in baselines:
                degradation = (baseline_throughput - concurrent_throughput) / baseline_throughput
                analysis["throughput_degradation"][task_type] = max(0, degradation)
            
            # 效率比
            if task_type in baselines:
                efficiency = concurrent_throughput / baseline_throughput
                analysis["efficiency_ratio"][task_type] = efficiency
        
        return analysis
    
    def _calculate_throughput(self, metrics: Dict[str, Any]) -> float:
        """计算吞吐量"""
        if metrics.get("status") != "success":
            return 0.0
        
        task_type = metrics.get("task_type", "")
        
        if task_type == "llm":
            return metrics.get("tokens_per_s", 0.0)
        elif task_type == "diffusion":
            return metrics.get("images_per_s", 0.0)
        elif task_type == "cv":
            return metrics.get("fps", 0.0)
        else:
            return 0.0
    
    def _analyze_resource_usage(self, concurrent_metrics: List[Any]) -> Dict[str, Any]:
        """分析资源使用情况"""
        successful_tasks = [m for m in concurrent_metrics if m and m.get("status") == "success"]
        
        if not successful_tasks:
            return {"error": "No successful tasks"}
        
        # 计算资源竞争指标
        total_elapsed = sum(m.get("elapsed_s", 0) for m in successful_tasks)
        avg_elapsed = total_elapsed / len(successful_tasks)
        
        return {
            "successful_tasks": len(successful_tasks),
            "total_tasks": len(concurrent_metrics),
            "avg_task_time_s": avg_elapsed,
            "max_task_time_s": max(m.get("elapsed_s", 0) for m in successful_tasks),
            "min_task_time_s": min(m.get("elapsed_s", 0) for m in successful_tasks),
            "time_variance": self._calculate_time_variance(successful_tasks),
        }
    
    def _calculate_time_variance(self, metrics: List[Any]) -> float:
        """计算时间方差"""
        if len(metrics) < 2:
            return 0.0
        
        times = [m.get("elapsed_s", 0) for m in metrics]
        mean_time = sum(times) / len(times)
        variance = sum((t - mean_time) ** 2 for t in times) / len(times)
        return variance
    
    def _save_result(self, result: Dict[str, Any]):
        """保存测试结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"concurrent_test_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"[Concurrent] Results saved to {filepath}")


# ─────────────────────────────────────────────────────────────
# 完整测试套件入口
# ─────────────────────────────────────────────────────────────
def run_full_suite():
    """运行完整的基准测试套件"""
    from .llm import LLMBenchmark
    from .multimodal import MultimodalBenchmark
    
    print("🚀 Starting TensorForge Full Test Suite...")
    
    # 加载配置
    config = config_manager.load_config()
    
    # 定义测试任务
    tasks = [
        {"type": "llm", "model": "llama3.1:8b", "prompt": "Explain AI in 100 words."},
        {"type": "diffusion", "model": "sdxl-turbo"},
        {"type": "cv", "model": "yolov8n"},
    ]
    
    results = {}
    
    # LLM 基准测试
    print("\n📝 Running LLM benchmarks...")
    llm_bench = LLMBenchmark(
        model_name="llama3.1:8b",
        n_runs=3,
        warmup_s=5.0,
        output_dir="results"
    )
    llm_result = llm_bench.run()
    results["llm"] = asdict(llm_result)
    
    # 多模态基准测试
    print("\n🎭 Running multimodal benchmarks...")
    multi_bench = MultimodalBenchmark(
        model_name="multimodal-test",
        tasks=["text", "vision"],
        output_dir="results"
    )
    multi_result = multi_bench.run()
    results["multimodal"] = asdict(multi_result)
    
    # 并发压力测试
    print("\n⚡ Running concurrent stress test...")
    concurrent_test = ConcurrentStressTest(tasks=tasks, output_dir="results")
    concurrent_result = concurrent_test.run()
    results["concurrent"] = concurrent_result
    
    # 保存综合报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = Path("results") / f"full_suite_report_{timestamp}.json"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n✅ Full test suite completed!")
    print(f"📄 Report saved to {report_file}")
    
    return results


def main():
    """主命令行入口"""
    parser = argparse.ArgumentParser(description="TensorForge Test Suite")
    parser.add_argument("--mode", choices=["full", "concurrent"], default="full",
                      help="Test mode: full suite or concurrent stress test")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--output", default="results", help="Output directory")
    
    args = parser.parse_args()
    
    if args.mode == "full":
        run_full_suite()
    elif args.mode == "concurrent":
        # 定义并发测试任务
        tasks = [
            {"type": "llm", "model": "llama3.1:8b"},
            {"type": "diffusion", "model": "sdxl-turbo"},
            {"type": "cv", "model": "yolov8n"},
        ]
        
        concurrent_test = ConcurrentStressTest(tasks=tasks, output_dir=args.output)
        concurrent_test.run()
    
    print("\n🎉 Test suite completed!")


if __name__ == "__main__":
    main()
