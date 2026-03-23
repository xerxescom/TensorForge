import pytest

import error_handler
from config_manager import BenchmarkConfig
from error_handler import RetryStrategy, retry_on_error


def test_retry_on_error_retries_and_raises():
    attempts = {"count": 0}

    @retry_on_error(RetryStrategy(max_retries=2, base_delay=0, backoff_factor=1))
    def always_fail():
        attempts["count"] += 1
        raise TimeoutError("timed out")

    with pytest.raises(TimeoutError):
        always_fail()

    assert attempts["count"] == 3


def test_retry_on_error_uses_config_defaults_and_reports_errors():
    attempts = {"count": 0}
    error_handler.config_manager._config = BenchmarkConfig(
        error_enable_retries=True,
        error_max_retries=1,
        error_retry_backoff_factor=1.0,
        error_report_errors=True,
    )
    error_handler.error_reporter.clear()

    @retry_on_error()
    def always_fail():
        attempts["count"] += 1
        raise TimeoutError("timed out")

    with pytest.raises(TimeoutError):
        always_fail()

    assert attempts["count"] == 2
    assert error_handler.error_reporter.error_counts[error_handler.ErrorType.TIMEOUT] == 2
