#!/usr/bin/env python3
"""
TensorForge 命令行接口
"""
import argparse
import sys

from .benchmarks.llm import LLMBenchmark
from .core.collector import GPUSampler
from .core.config_manager import config_manager


def run_llm_benchmark(args):
    """运行 LLM 基准测试"""
    print(f"🚀 运行 LLM 基准测试: {args.model}")
    
    bench = LLMBenchmark(
        model_name=args.model,
        n_runs=args.runs,
        warmup_s=args.warmup,
        timeout_s=args.timeout,
        output_dir=args.output
    )
    
    result = bench.run()
    
    print(f"\n✅ 测试完成!")
    print(f"   状态: {result.status}")
    print(f"   耗时: {result.duration_s:.2f}s")
    
    if result.metrics:
        print(f"   Tokens/s: {result.metrics.get('tokens_per_s_mean', 0):.1f}")
        print(f"   成功率: {result.metrics.get('success_rate', 0):.1%}")
        print(f"   TTFT: {result.metrics.get('ttft_s_mean', 0):.3f}s")
    
    if result.gpu_stats:
        print(f"   GPU 利用率: {result.gpu_stats.get('gpu_util_mean', 0):.1f}%")
        print(f"   功耗: {result.gpu_stats.get('power_mean', 0):.1f}W")


def run_gpu_monitor(args):
    """运行 GPU 监控"""
    print(f"🔍 GPU 监控 (运行 {args.duration}s)...")
    
    sampler = GPUSampler(interval_s=args.interval)
    
    try:
        sampler.start()
        
        import time
        time.sleep(args.duration)
        
        samples = sampler.stop()
        stats = sampler.get_stats()
        
        print(f"\n📊 监控结果:")
        print(f"   样本数量: {len(samples)}")
        print(f"   GPU 利用率: {stats.get('gpu_util_mean', 0):.1f}% (±{stats.get('gpu_util_max', 0) - stats.get('gpu_util_mean', 0):.1f})")
        print(f"   功耗: {stats.get('power_mean', 0):.1f}W")
        print(f"   温度: {stats.get('temp_mean', 0):.1f}°C")
        print(f"   查询成功率: {(1-stats.get('error_rate', 0)):.1%}")
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 监控失败: {e}")


def show_config(args):
    """显示配置"""
    print("⚙️  当前配置:")
    
    config = config_manager.load_config()
    
    print(f"   采样间隔: {config.sample_interval_s}s")
    print(f"   预热时间: {config.warmup_s}s")
    print(f"   超时时间: {config.timeout_s}s")
    print(f"   智能采样: {config.adaptive_sampling}")
    print(f"   GPU 高利用率阈值: {config.high_utilization}%")
    print(f"   LLM 提示词数量: {len(config.llm_prompts)}")


def main():
    """主命令行入口"""
    parser = argparse.ArgumentParser(
        prog="tensorforge",
        description="TensorForge - GPU AI Benchmark Suite"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # LLM 基准测试命令
    llm_parser = subparsers.add_parser("llm", help="运行 LLM 基准测试")
    llm_parser.add_argument("--model", default="llama3.1:8b", help="模型名称")
    llm_parser.add_argument("--runs", type=int, default=5, help="运行次数")
    llm_parser.add_argument("--warmup", type=float, default=5.0, help="预热时间(秒)")
    llm_parser.add_argument("--timeout", type=int, default=120, help="超时时间(秒)")
    llm_parser.add_argument("--output", default="results", help="输出目录")
    llm_parser.set_defaults(func=run_llm_benchmark)
    
    # GPU 监控命令
    monitor_parser = subparsers.add_parser("monitor", help="GPU 监控")
    monitor_parser.add_argument("--duration", type=int, default=30, help="监控时长(秒)")
    monitor_parser.add_argument("--interval", type=float, default=1.0, help="采样间隔(秒)")
    monitor_parser.set_defaults(func=run_gpu_monitor)
    
    # 配置显示命令
    config_parser = subparsers.add_parser("config", help="显示配置")
    config_parser.set_defaults(func=show_config)
    
    # 解析参数
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
