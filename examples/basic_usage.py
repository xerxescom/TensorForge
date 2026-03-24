#!/usr/bin/env python3
"""
TensorForge 基础使用示例
"""
import sys
from pathlib import Path

# 添加 src 目录到路径
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from src.tensorforge import LLMBenchmark, GPUSampler, config_manager

def basic_llm_benchmark():
    """基础 LLM 基准测试示例"""
    print("🚀 运行基础 LLM 基准测试...")
    
    # 创建基准测试
    bench = LLMBenchmark(
        model_name="llama3.1:8b",
        n_runs=3,
        warmup_s=5.0,
        output_dir="results"
    )
    
    # 运行测试
    result = bench.run()
    
    # 显示结果
    print(f"✅ 测试完成!")
    print(f"   状态: {result.status}")
    print(f"   耗时: {result.duration_s:.2f}s")
    if result.metrics:
        print(f"   Tokens/s: {result.metrics.get('tokens_per_s_mean', 0):.1f}")
        print(f"   成功率: {result.metrics.get('success_rate', 0):.1%}")

def gpu_sampling_demo():
    """GPU 采样演示"""
    print("🔍 GPU 采样演示...")
    
    # 创建采样器
    sampler = GPUSampler(interval_s=1.0, adaptive_sampling=True)
    
    try:
        # 启动采样
        sampler.start()
        print("   采样已启动，运行 5 秒...")
        
        import time
        time.sleep(5)
        
        # 停止采样
        samples = sampler.stop()
        print(f"   收集到 {len(samples)} 个样本")
        
        # 显示统计
        stats = sampler.get_stats()
        print(f"   GPU 利用率: {stats.get('gpu_util_mean', 0):.1f}%")
        print(f"   查询成功率: {(1-stats.get('error_rate', 0)):.1%}")
        
    except Exception as e:
        print(f"   ❌ GPU 采样失败: {e}")

def config_demo():
    """配置管理演示"""
    print("⚙️  配置管理演示...")
    
    # 加载配置
    config = config_manager.load_config()
    
    print(f"   采样间隔: {config.sample_interval_s}s")
    print(f"   预热时间: {config.warmup_s}s")
    print(f"   超时时间: {config.timeout_s}s")
    print(f"   LLM 提示词数量: {len(config.llm_prompts)}")

if __name__ == "__main__":
    print("TensorForge 基础使用示例")
    print("=" * 40)
    
    try:
        config_demo()
        print()
        gpu_sampling_demo()
        print()
        basic_llm_benchmark()
    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 运行错误: {e}")
