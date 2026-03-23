"""
增强的错误处理和重试机制
"""
from enum import Enum
from typing import Optional, Dict, Any
import time
import functools

from config_manager import config_manager
from tf_logger import logger


class ErrorType(Enum):
    """错误类型分类"""
    TIMEOUT = "timeout"
    OUT_OF_MEMORY = "out_of_memory"
    DEPENDENCY_MISSING = "dependency_missing"
    GPU_DRIVER_ERROR = "gpu_driver_error"
    NETWORK_ERROR = "network_error"
    PERMISSION_ERROR = "permission_error"
    UNKNOWN = "unknown"


class RetryStrategy:
    """重试策略配置"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor
    
    def get_delay(self, attempt: int) -> float:
        """计算重试延迟（指数退避）"""
        return self.base_delay * (self.backoff_factor ** attempt)
    
    def should_retry(self, error_type: ErrorType) -> bool:
        """判断是否应该重试"""
        no_retry_types = {ErrorType.PERMISSION_ERROR, ErrorType.DEPENDENCY_MISSING}
        return error_type not in no_retry_types


class ErrorClassifier:
    """错误分类器"""
    
    @staticmethod
    def classify(exception: Exception) -> tuple[ErrorType, str]:
        """分类异常并返回错误类型和描述"""
        error_msg = str(exception).lower()
        exception_type = type(exception).__name__
        logger.debug(f"[error] Classifying exception: {exception_type}: {error_msg}")
        
        if "timeout" in error_msg or "timed out" in error_msg:
            logger.debug(f"[error] Classified as TIMEOUT: {exception_type}")
            return ErrorType.TIMEOUT, "Operation timed out"
        elif "out of memory" in error_msg or "cuda out of memory" in error_msg:
            logger.debug(f"[error] Classified as OUT_OF_MEMORY: {exception_type}")
            return ErrorType.OUT_OF_MEMORY, "GPU memory exhausted"
        elif "no module named" in error_msg or "not found" in error_msg:
            logger.debug(f"[error] Classified as DEPENDENCY_MISSING: {exception_type}")
            return ErrorType.DEPENDENCY_MISSING, "Required dependency missing"
        elif "nvidia" in error_msg or "cuda" in error_msg:
            logger.debug(f"[error] Classified as GPU_DRIVER_ERROR: {exception_type}")
            return ErrorType.GPU_DRIVER_ERROR, "GPU driver or CUDA error"
        elif "permission denied" in error_msg or "access denied" in error_msg:
            logger.debug(f"[error] Classified as PERMISSION_ERROR: {exception_type}")
            return ErrorType.PERMISSION_ERROR, "Permission denied"
        elif "connection" in error_msg or "network" in error_msg:
            logger.debug(f"[error] Classified as NETWORK_ERROR: {exception_type}")
            return ErrorType.NETWORK_ERROR, "Network connection error"
        else:
            logger.debug(f"[error] Classified as UNKNOWN: {exception_type}")
            return ErrorType.UNKNOWN, f"Unknown error: {str(exception)}"


def retry_on_error(strategy: Optional[RetryStrategy] = None):
    """重试装饰器"""
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
            logger.debug(f"[error] Entering retry wrapper for {func.__name__} (max_retries={strategy.max_retries})")
            last_exception = None
            last_error_type = ErrorType.UNKNOWN
            last_error_desc = "Unknown error"
            
            for attempt in range(strategy.max_retries + 1):
                try:
                    if attempt > 0:
                        logger.debug(f"[error] Retry attempt {attempt}/{strategy.max_retries} for {func.__name__}")
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"[error] {func.__name__} succeeded on attempt {attempt + 1}")
                    return result
                except Exception as e:
                    error_type, error_desc = ErrorClassifier.classify(e)
                    last_exception = e
                    last_error_type = error_type
                    last_error_desc = error_desc
                    
                    logger.warning(f"[error] {func.__name__} failed on attempt {attempt + 1}: {error_desc} ({error_type.value})")
                    
                    config = config_manager.load_config()
                    if config.error_report_errors:
                        logger.debug(f"[error] Reporting error for {func.__name__}: {error_type.value}")
                        error_reporter.report_error(
                            error_type,
                            error_desc,
                            context={"function": func.__name__, "attempt": attempt + 1},
                        )
                    
                    if attempt == strategy.max_retries or not strategy.should_retry(error_type):
                        logger.debug(f"[error] No more retries for {func.__name__} (attempt={attempt}, retryable={strategy.should_retry(error_type)})")
                        break
                    
                    delay = strategy.get_delay(attempt)
                    logger.warning(
                        f"[error] {func.__name__} failed ({error_desc}), retrying in {delay}s..."
                    )
                    time.sleep(delay)
            
            # 所有重试都失败了
            logger.error(
                f"[error] {func.__name__} failed after {strategy.max_retries} retries "
                f"({last_error_type.value}): {last_error_desc}"
            )
            logger.debug(f"[error] Raising original exception from {func.__name__}: {type(last_exception).__name__}")
            raise last_exception
        
        return wrapper
    return decorator


class ErrorReporter:
    """错误报告器"""
    
    def __init__(self):
        self.error_counts: Dict[ErrorType, int] = {}
        self.error_details: Dict[ErrorType, list] = {}
    
    def report_error(self, error_type: ErrorType, details: str, context: Dict[str, Any] = None):
        """报告错误"""
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        count = self.error_counts[error_type]
        
        if error_type not in self.error_details:
            self.error_details[error_type] = []
        
        error_entry = {
            'details': details,
            'context': context or {},
            'timestamp': time.time()
        }
        self.error_details[error_type].append(error_entry)
        
        logger.debug(f"[error] Reported {error_type.value} error (count={count}): {details}")
        if context:
            logger.debug(f"[error] Error context: {context}")
    
    def get_summary(self) -> Dict[str, Any]:
        """获取错误摘要"""
        total_errors = sum(self.error_counts.values())
        most_common = max(self.error_counts.items(), key=lambda x: x[1]) if self.error_counts else None
        
        logger.debug(f"[error] Error summary: {total_errors} total errors")
        if most_common:
            logger.debug(f"[error] Most common error: {most_common[0].value} ({most_common[1]} occurrences)")
        
        return {
            'total_errors': total_errors,
            'error_counts': {e.value: c for e, c in self.error_counts.items()},
            'most_common': most_common
        }
    
    def clear(self):
        """清除错误记录"""
        total_cleared = sum(self.error_counts.values())
        self.error_counts.clear()
        self.error_details.clear()
        logger.info(f"[error] Cleared {total_cleared} error records from reporter")


# 全局错误报告器
error_reporter = ErrorReporter()
