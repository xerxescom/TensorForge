"""
TensorForge 日志配置
"""
import sys
from pathlib import Path

from tf_logger import logger


def configure_logging(output_dir: str | None = None, verbose: bool = True):
    """配置统一日志输出。"""
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if verbose else "INFO",
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )

    if output_dir:
        log_path = Path(output_dir) / "tensorforge.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_path,
            level="DEBUG",
            encoding="utf-8",
            rotation="10 MB",
            retention=5,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        )

    return logger
