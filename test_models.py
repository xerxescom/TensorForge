#!/usr/bin/env python3
"""
模型下载和测试脚本
测试新的模型管理系统
"""
import sys
import time
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from model_manager import model_manager
from network_optimizer import download_optimizer
from other_bench import DiffusionBenchmark
from llm_bench import LLMBenchmark


def test_model_download():
    """测试模型下载"""
    print("=" * 60)
    print("Testing Model Download System")
    print("=" * 60)
    
    # 设置网络优化
    download_optimizer.setup()
    
    try:
        # 测试 SDXL-Turbo 下载
        print("\n1. Testing SDXL-Turbo download...")
        start_time = time.time()
        
        try:
            model_path = model_manager.get_model_path("sdxl-turbo")
            download_time = time.time() - start_time
            print(f"✓ SDXL-Turbo downloaded successfully in {download_time:.2f}s")
            print(f"  Path: {model_path}")
            
            # 检查文件
            model_dir = Path(model_path)
            if model_dir.exists():
                files = list(model_dir.rglob("*"))
                print(f"  Files: {len(files)} items")
                
        except Exception as e:
            print(f"✗ SDXL-Turbo download failed: {e}")
        
        # 测试 YOLOv8n 下载
        print("\n2. Testing YOLOv8n download...")
        start_time = time.time()
        
        try:
            model_path = model_manager.get_model_path("yolov8n")
            download_time = time.time() - start_time
            print(f"✓ YOLOv8n downloaded successfully in {download_time:.2f}s")
            print(f"  Path: {model_path}")
            
        except Exception as e:
            print(f"✗ YOLOv8n download failed: {e}")
        
        # 测试 Whisper Base 下载
        print("\n3. Testing Whisper Base download...")
        start_time = time.time()
        
        try:
            model_path = model_manager.get_model_path("whisper-base")
            download_time = time.time() - start_time
            print(f"✓ Whisper Base downloaded successfully in {download_time:.2f}s")
            print(f"  Path: {model_path}")
            
        except Exception as e:
            print(f"✗ Whisper Base download failed: {e}")
    
    finally:
        download_optimizer.cleanup()


def test_diffusion_benchmark():
    """测试 Diffusion 基准测试"""
    print("\n" + "=" * 60)
    print("Testing Diffusion Benchmark with New System")
    print("=" * 60)
    
    try:
        # 创建基准测试实例
        bench = DiffusionBenchmark(
            model_name="sdxl-turbo",  # 使用简化的模型名
            n_images=2,  # 减少测试数量
            n_steps=4,   # 减少步数
            output_dir="test_results"
        )
        
        print("Running diffusion benchmark...")
        result = bench.run()
        
        print("✓ Diffusion benchmark completed")
        print(f"  Duration: {result.duration_s}s")
        print(f"  Status: {result.status}")
        
        if result.metrics:
            print(f"  Images per second: {result.metrics.get('it_per_s', 'N/A')}")
            print(f"  Success rate: {result.metrics.get('success_rate', 'N/A')}")
        
    except Exception as e:
        print(f"✗ Diffusion benchmark failed: {e}")


def test_llm_benchmark():
    """测试 LLM 基准测试"""
    print("\n" + "=" * 60)
    print("Testing LLM Benchmark with New System")
    print("=" * 60)
    
    try:
        # 创建基准测试实例
        bench = LLMBenchmark(
            model_name="llama3.1:8b",
            n_runs=2,  # 减少测试数量
            output_dir="test_results"
        )
        
        print("Running LLM benchmark...")
        result = bench.run()
        
        print("✓ LLM benchmark completed")
        print(f"  Duration: {result.duration_s}s")
        print(f"  Status: {result.status}")
        
        if result.metrics:
            print(f"  Tokens/s: {result.metrics.get('tokens_per_s_mean', 'N/A')}")
            print(f"  Success rate: {result.metrics.get('success_rate', 'N/A')}")
        
    except Exception as e:
        print(f"✗ LLM benchmark failed: {e}")


def show_cache_info():
    """显示缓存信息"""
    print("\n" + "=" * 60)
    print("Cache Information")
    print("=" * 60)
    
    cache_dir = Path("models_cache")
    if cache_dir.exists():
        total_size = 0
        model_dirs = [d for d in cache_dir.iterdir() if d.is_dir()]
        
        print(f"Cache directory: {cache_dir}")
        print(f"Models cached: {len(model_dirs)}")
        
        for model_dir in model_dirs:
            size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
            total_size += size
            size_mb = size / (1024 * 1024)
            print(f"  {model_dir.name}: {size_mb:.1f} MB")
        
        total_mb = total_size / (1024 * 1024)
        print(f"Total cache size: {total_mb:.1f} MB")
    else:
        print("No cache directory found")


def main():
    """主函数"""
    print("TensorForge Model System Test")
    print("This script tests the new model download and caching system")
    
    # 显示缓存信息
    show_cache_info()
    
    # 测试模型下载
    test_model_download()
    
    # 更新缓存信息
    show_cache_info()
    
    # 测试基准测试（可选）
    test_choice = input("\nRun benchmark tests? (y/n): ").lower().strip()
    if test_choice == 'y':
        test_diffusion_benchmark()
        test_llm_benchmark()
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
