import pytest

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
