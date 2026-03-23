"""
增强的错误处理和重试机制
"""
from enum import Enum
from typing import Optional, Dict, Any
import time
import functools


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
        
        if "timeout" in error_msg or "timed out" in error_msg:
            return ErrorType.TIMEOUT, "Operation timed out"
        elif "out of memory" in error_msg or "cuda out of memory" in error_msg:
            return ErrorType.OUT_OF_MEMORY, "GPU memory exhausted"
        elif "no module named" in error_msg or "not found" in error_msg:
            return ErrorType.DEPENDENCY_MISSING, "Required dependency missing"
        elif "nvidia" in error_msg or "cuda" in error_msg:
            return ErrorType.GPU_DRIVER_ERROR, "GPU driver or CUDA error"
        elif "permission denied" in error_msg or "access denied" in error_msg:
            return ErrorType.PERMISSION_ERROR, "Permission denied"
        elif "connection" in error_msg or "network" in error_msg:
            return ErrorType.NETWORK_ERROR, "Network connection error"
        else:
            return ErrorType.UNKNOWN, f"Unknown error: {str(exception)}"


def retry_on_error(strategy: Optional[RetryStrategy] = None):
    """重试装饰器"""
    if strategy is None:
        strategy = RetryStrategy()
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            last_error_type = ErrorType.UNKNOWN
            last_error_desc = "Unknown error"
            
            for attempt in range(strategy.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_type, error_desc = ErrorClassifier.classify(e)
                    last_exception = e
                    last_error_type = error_type
                    last_error_desc = error_desc
                    
                    if attempt == strategy.max_retries or not strategy.should_retry(error_type):
                        break
                    
                    delay = strategy.get_delay(attempt)
                    print(f"[retry] {func.__name__} failed ({error_desc}), retrying in {delay}s...")
                    time.sleep(delay)
            
            # 所有重试都失败了
            print(
                f"[error] {func.__name__} failed after {strategy.max_retries} retries "
                f"({last_error_type.value}): {last_error_desc}"
            )
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
        
        if error_type not in self.error_details:
            self.error_details[error_type] = []
        
        self.error_details[error_type].append({
            'details': details,
            'context': context or {},
            'timestamp': time.time()
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """获取错误摘要"""
        return {
            'total_errors': sum(self.error_counts.values()),
            'error_counts': {e.value: c for e, c in self.error_counts.items()},
            'most_common': max(self.error_counts.items(), key=lambda x: x[1]) if self.error_counts else None
        }
    
    def clear(self):
        """清除错误记录"""
        self.error_counts.clear()
        self.error_details.clear()


# 全局错误报告器
error_reporter = ErrorReporter()
