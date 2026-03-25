"""
TensorForge - GPU AI Benchmark Suite
全自动 GPU AI 能力测试框架
"""

__version__ = "1.0.0"
__author__ = "TensorForge Team"
__email__ = "team@tensorforge.dev"

# 导入核心组件
from .core.collector import GPUSampler, BenchmarkRunner, GPUSample, BenchmarkResult
from .core.config_manager import ConfigManager, BenchmarkConfig, config_manager
from .core.error_handler import ErrorHandler, ErrorType, RetryStrategy

# 导入基准测试
from .benchmarks.llm import LLMBenchmark, LLMContextScaleBenchmark
from .benchmarks.multimodal import MultimodalBenchmark, CrossModalBenchmark
from .benchmarks.suite import ConcurrentStressTest, run_full_suite

# 公开 API
__all__ = [
    # Core
    "GPUSampler",
    "BenchmarkRunner",
    "GPUSample",
    "BenchmarkResult",
    # Configuration
    "ConfigManager",
    "BenchmarkConfig",
    "config_manager",
    # Error Handling
    "ErrorHandler",
    "ErrorType",
    "RetryStrategy",
    # Benchmarks
    "LLMBenchmark",
    "LLMContextScaleBenchmark",
    "MultimodalBenchmark",
    "CrossModalBenchmark",
    "ConcurrentStressTest",
    "run_full_suite",
]