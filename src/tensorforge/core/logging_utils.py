"""
日志工具模块
"""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "tensorforge",
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径，None 表示不写入文件
        format_string: 日志格式字符串
    
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # 默认格式
    if format_string is None:
        format_string = "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s"
    
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "tensorforge") -> logging.Logger:
    """获取日志记录器"""
    return logging.getLogger(name)


def set_log_level(logger: logging.Logger, level: str):
    """设置日志级别"""
    logger.setLevel(getattr(logging, level.upper()))


# 默认日志记录器
default_logger = setup_logger()
