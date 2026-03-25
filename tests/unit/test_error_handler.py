"""单元测试 - core/error_handler.py"""

import pytest

from src.core.error_handler import ErrorType, RetryStrategy, ErrorHandler


class TestErrorType:
    def test_error_types_exist(self):
        assert ErrorType.NETWORK
        assert ErrorType.SYSTEM
        assert ErrorType.MODEL
        assert ErrorType.CONFIG
        assert ErrorType.TIMEOUT
        assert ErrorType.UNKNOWN


class TestRetryStrategy:
    def test_default_strategy(self):
        strategy = RetryStrategy()
        assert strategy.max_attempts == 3
        assert strategy.base_delay == 1.0
        assert strategy.exponential_base == 2.0
        assert strategy.jitter is True


class TestErrorHandler:
    def test_error_handler_creation(self):
        strategy = RetryStrategy(max_attempts=2)
        handler = ErrorHandler(strategy)
        assert handler.strategy.max_attempts == 2
