"""
Enhanced error handling and retry mechanisms.
"""

from __future__ import annotations

import functools
import time
from enum import Enum
from typing import Any

from .config_manager import config_manager
from .tf_logger import logger


class ErrorType(Enum):
    TIMEOUT = "timeout"
    OUT_OF_MEMORY = "out_of_memory"
    DEPENDENCY_MISSING = "dependency_missing"
    GPU_DRIVER_ERROR = "gpu_driver_error"
    NETWORK_ERROR = "network_error"
    PERMISSION_ERROR = "permission_error"
    UNKNOWN = "unknown"


class RetryStrategy:
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor

    def get_delay(self, attempt: int) -> float:
        return self.base_delay * (self.backoff_factor**attempt)

    def should_retry(self, error_type: ErrorType) -> bool:
        no_retry_types = {ErrorType.PERMISSION_ERROR, ErrorType.DEPENDENCY_MISSING}
        return error_type not in no_retry_types


class ErrorClassifier:
    @staticmethod
    def classify(exception: Exception) -> tuple[ErrorType, str]:
        error_msg = str(exception).lower()
        exception_type = type(exception).__name__
        logger.debug(f"[error] Classifying exception: {exception_type}: {error_msg}")

        if "timeout" in error_msg or "timed out" in error_msg:
            return ErrorType.TIMEOUT, "Operation timed out"
        if "out of memory" in error_msg or "cuda out of memory" in error_msg:
            return ErrorType.OUT_OF_MEMORY, "GPU memory exhausted"
        if "no module named" in error_msg or "not found" in error_msg:
            return ErrorType.DEPENDENCY_MISSING, "Required dependency missing"
        if "nvidia" in error_msg or "cuda" in error_msg:
            return ErrorType.GPU_DRIVER_ERROR, "GPU driver or CUDA error"
        if "permission denied" in error_msg or "access denied" in error_msg:
            return ErrorType.PERMISSION_ERROR, "Permission denied"
        if "connection" in error_msg or "network" in error_msg:
            return ErrorType.NETWORK_ERROR, "Network connection error"
        return ErrorType.UNKNOWN, f"Unknown error: {exception!s}"


def retry_on_error(strategy: RetryStrategy | None = None):
    if strategy is None:
        config = config_manager.load_config()
        strategy = RetryStrategy(
            max_retries=config.error_max_retries if config.error_enable_retries else 0,
            base_delay=1.0,
            backoff_factor=config.error_retry_backoff_factor,
        )

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception: Exception | None = None
            last_error_type = ErrorType.UNKNOWN
            last_error_desc = "Unknown error"

            for attempt in range(strategy.max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"[error] {func.__name__} succeeded on attempt {attempt + 1}")
                    return result
                except Exception as e:
                    error_type, error_desc = ErrorClassifier.classify(e)
                    last_exception = e
                    last_error_type = error_type
                    last_error_desc = error_desc

                    logger.warning(
                        f"[error] {func.__name__} failed on attempt {attempt + 1}: "
                        f"{error_desc} ({error_type.value})"
                    )

                    config = config_manager.load_config()
                    if config.error_report_errors:
                        error_reporter.report_error(
                            error_type,
                            error_desc,
                            context={"function": func.__name__, "attempt": attempt + 1},
                        )

                    if attempt == strategy.max_retries or not strategy.should_retry(error_type):
                        break

                    delay = strategy.get_delay(attempt)
                    logger.warning(f"[error] {func.__name__} retrying in {delay}s...")
                    time.sleep(delay)

            logger.error(
                f"[error] {func.__name__} failed after {strategy.max_retries} retries "
                f"({last_error_type.value}): {last_error_desc}"
            )
            assert last_exception is not None
            raise last_exception

        return wrapper

    return decorator


class ErrorReporter:
    def __init__(self):
        self.error_counts: dict[ErrorType, int] = {}
        self.error_details: dict[ErrorType, list] = {}

    def report_error(
        self, error_type: ErrorType, details: str, context: dict[str, Any] | None = None
    ):
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        if error_type not in self.error_details:
            self.error_details[error_type] = []
        self.error_details[error_type].append(
            {
                "details": details,
                "context": context or {},
                "timestamp": time.time(),
            }
        )

    def get_summary(self) -> dict[str, Any]:
        total_errors = sum(self.error_counts.values())
        most_common = (
            max(self.error_counts.items(), key=lambda x: x[1]) if self.error_counts else None
        )
        return {
            "total_errors": total_errors,
            "error_counts": {e.value: c for e, c in self.error_counts.items()},
            "most_common": most_common,
        }

    def clear(self):
        self.error_counts.clear()
        self.error_details.clear()
        logger.info("[error] Cleared error records from reporter")


error_reporter = ErrorReporter()
