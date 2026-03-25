"""单元测试 - core/config_manager.py"""

import pytest

from src.core.config_manager import BenchmarkConfig, ConfigManager


class TestBenchmarkConfig:
    def test_default_values(self):
        config = BenchmarkConfig()
        assert config.sample_interval_s == 0.5
        assert config.warmup_s == 5.0
        assert config.timeout_s == 120

    def test_custom_values(self):
        config = BenchmarkConfig(
            sample_interval_s=1.0,
            warmup_s=10.0,
            timeout_s=300,
        )
        assert config.sample_interval_s == 1.0
        assert config.warmup_s == 10.0


class TestConfigManager:
    def test_singleton(self):
        from src.core.config_manager import config_manager as cm1
        from src.core.config_manager import config_manager as cm2
        assert cm1 is cm2
