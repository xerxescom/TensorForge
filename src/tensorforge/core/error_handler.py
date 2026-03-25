"""
错误处理和重试机制
"""
import time
from enum import Enum
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass


class ErrorType(Enum):
    """错误类型枚举"""
    NETWORK = "network"
    SYSTEM = "system"
    MODEL = "model"
    CONFIG = "config"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class RetryStrategy:
    """重试策略配置"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


class ErrorHandler:
    """统一错误处理器"""
    
    def __init__(self, retry_strategy: Optional[RetryStrategy] = None):
        self.retry_strategy = retry_strategy or RetryStrategy()
        self.error_counts: Dict[str, int] = {}
    
    def classify_error(self, error: Exception) -> ErrorType:
        """分类错误类型"""
        error_str = str(error).lower()
        
        if "timeout" in error_str or "timed out" in error_str:
            return ErrorType.TIMEOUT
        elif "connection" in error_str or "network" in error_str:
            return ErrorType.NETWORK
        elif "model" in error_str or "checkpoint" in error_str:
            return ErrorType.MODEL
        elif "config" in error_str or "yaml" in error_str or "json" in error_str:
            return ErrorType.CONFIG
        elif "cuda" in error_str or "gpu" in error_str or "nvidia" in error_str:
            return ErrorType.SYSTEM
        else:
            return ErrorType.UNKNOWN
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """判断是否应该重试"""
        if attempt >= self.retry_strategy.max_attempts:
            return False
        
        error_type = self.classify_error(error)
        
        # 某些错误类型不适合重试
        no_retry_types = [ErrorType.CONFIG, ErrorType.MODEL]
        if error_type in no_retry_types:
            return False
        
        return True
    
    def calculate_delay(self, attempt: int) -> float:
        """计算重试延迟"""
        delay = self.retry_strategy.base_delay * (
            self.retry_strategy.exponential_base ** (attempt - 1)
        )
        delay = min(delay, self.retry_strategy.max_delay)
        
        if self.retry_strategy.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)
        
        return delay
    
    def retry(self, func: Callable, *args, **kwargs) -> Any:
        """执行带重试的函数调用"""
        last_error = None
        
        for attempt in range(1, self.retry_strategy.max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                error_type = self.classify_error(e)
                
                # 记录错误统计
                error_key = f"{error_type.value}:{type(e).__name__}"
                self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
                
                if not self.should_retry(e, attempt):
                    break
                
                delay = self.calculate_delay(attempt)
                print(f"[retry] {error_type.value} error (attempt {attempt}/{self.retry_strategy.max_attempts}), retrying in {delay:.1f}s...")
                time.sleep(delay)
        
        raise last_error
    
    def get_error_stats(self) -> Dict[str, int]:
        """获取错误统计"""
        return self.error_counts.copy()
    
    def reset_stats(self):
        """重置错误统计"""
        self.error_counts.clear()


# 全局错误处理器实例
error_handler = ErrorHandler()
