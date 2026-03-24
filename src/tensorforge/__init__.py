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
from .benchmarks.llm import LLMBenchmark, LLMContextScaleBenchmark

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
    
    # Benchmarks
    "LLMBenchmark",
    "LLMContextScaleBenchmark",
]