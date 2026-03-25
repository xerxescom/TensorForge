"""测试配置和共享 fixtures"""

import pytest


@pytest.fixture
def mock_gpu_sample():
    """模拟 GPU 采样数据"""
    return {
        "timestamp": 1234567890.0,
        "gpu_util": 80.0,
        "memory_used_mb": 4096.0,
        "memory_total_mb": 8192.0,
        "power_w": 150.0,
        "temp_c": 70.0,
        "clock_mhz": 1800.0,
        "memory_clock_mhz": 6000.0,
    }


@pytest.fixture
def temp_output_dir(tmp_path):
    """临时输出目录"""
    return tmp_path / "test_results"
