#!/usr/bin/env python3
"""
模型测试工具
"""
import sys
from pathlib import Path

from ..benchmarks.llm import LLMBenchmark
from ..benchmarks.multimodal import MultimodalBenchmark
from ..core.config_manager import config_manager
from ..core.model_manager import model_manager


def test_model_download():
    """测试模型下载"""
    print("=" * 60)
    print("Testing Model Download System")
    print("=" * 60)

    try:
        # 测试模型路径获取
        print("\n1. Testing model path retrieval...")
        model_path = model_manager.get_model_path("test-model")
        if model_path:
            print(f"✓ Model path found: {model_path}")
        else:
            print("✗ Model path not found")

        # 显示缓存信息
        print("\n2. Cache information:")
        cached_models = model_manager.list_cached_models()
        if cached_models:
            print(f"✓ Cached models: {', '.join(cached_models)}")
        else:
            print("✗ No cached models")

        # 测试缓存清理
        print("\n3. Testing cache management...")
        cache_dir = Path("models_cache")
        if cache_dir.exists():
            total_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
            size_mb = total_size / (1024 * 1024)
            print(f"✓ Cache directory: {cache_dir}")
            print(f"✓ Total cache size: {size_mb:.1f} MB")
        else:
            print("✗ Cache directory not found")

        print("\n✓ Model management system test completed")

    except Exception as e:
        print(f"✗ Model management test failed: {e}")
        raise


def test_benchmark_integration():
    """测试基准测试集成"""
    print("\n" + "=" * 60)
    print("Testing Benchmark Integration")
    print("=" * 60)

    try:
        # 测试配置加载
        print("\n1. Testing configuration system...")
        config = config_manager.load_config()
        print(f"✓ Config loaded: sample_interval={config.sample_interval_s}s")

        # 测试 LLM 基准测试创建
        print("\n2. Testing LLM benchmark creation...")
        llm_bench = LLMBenchmark(
            model_name="test-model",
            n_runs=1,
            warmup_s=0,
            output_dir="test_results"
        )
        print("✓ LLM benchmark instance created successfully")

        # 测试多模态基准测试创建
        print("\n3. Testing multimodal benchmark creation...")
        multi_bench = MultimodalBenchmark(
            model_name="test-multimodal",
            tasks=["text"],
            output_dir="test_results"
        )
        print("✓ Multimodal benchmark instance created successfully")

        print("\n✓ Benchmark integration test completed")

    except Exception as e:
        print(f"✗ Benchmark integration test failed: {e}")
        raise


def test_import_system():
    """测试导入系统"""
    print("\n" + "=" * 60)
    print("Testing Import System")
    print("=" * 60)

    try:
        # 测试核心模块导入
        print("\n1. Testing core module imports...")
        from ..core.collector import GPUSampler, BenchmarkRunner
        from ..core.config_manager import config_manager
        from ..core.error_handler import error_handler
        from ..core.logging_utils import get_logger
        from ..core.model_manager import ModelManager
        from ..core.network_optimizer import NetworkOptimizer
        print("✓ All core modules imported successfully")

        # 测试基准测试模块导入
        print("\n2. Testing benchmark module imports...")
        from ..benchmarks.llm import LLMBenchmark
        from ..benchmarks.multimodal import MultimodalBenchmark
        print("✓ All benchmark modules imported successfully")

        # 测试主包导入
        print("\n3. Testing main package import...")
        import src
        print("✓ Main package imported successfully")

        print("\n✓ Import system test completed")

    except Exception as e:
        print(f"✗ Import system test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """主函数"""
    print("TensorForge Model System Test")
    print("This script tests the new model and benchmark systems")
    print()

    tests = [
        ("Import System", test_import_system),
        ("Model Management", test_model_download),
        ("Benchmark Integration", test_benchmark_integration),
    ]

    passed = 0
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            test_func()
            passed += 1
            print(f"✓ {test_name}: PASSED")
        except Exception as e:
            print(f"✗ {test_name}: FAILED - {e}")

    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{len(tests)} test groups passed")

    if passed == len(tests):
        print("🎉 All tests passed! System is ready for use.")
        print("\nNext steps:")
        print("1. Run benchmarks: python -m src.cli llm")
        print("2. Run tests: python -m pytest")
        print("3. Check documentation: README.md")
    else:
        print("⚠️  Some tests failed. Check the errors above.")

    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
