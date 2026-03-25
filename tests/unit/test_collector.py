"""单元测试 - core/collector.py"""

import pytest

from src.core.collector import GPUSample, GPUSampler


class TestGPUSample:
    def test_dataclass_creation(self):
        sample = GPUSample(
            timestamp=1234567890.0,
            gpu_util=80.0,
            memory_used_mb=4096.0,
            memory_total_mb=8192.0,
            power_w=150.0,
            temp_c=70.0,
            clock_mhz=1800.0,
            memory_clock_mhz=6000.0,
        )
        assert sample.gpu_util == 80.0
        assert sample.memory_used_mb == 4096.0


class TestGPUSampler:
    def test_get_stats_empty(self):
        sampler = GPUSampler(interval_s=0.5, gpu_index=0)
        stats = sampler.get_stats()
        assert stats["mean_util"] == 0.0
        assert stats["mean_power"] == 0.0
