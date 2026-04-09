"""
TensorForge logger facade.

Prefers the third-party `loguru` package when available. If a local `loguru.py`
module is shadowing the third-party package (common in repo roots), we treat
that as "not available" and fall back to the built-in logger.
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO


def _has_real_loguru() -> bool:
    spec = importlib.util.find_spec("loguru")
    if spec is None or spec.origin is None:
        return False
    # If the resolved module is a local file named `loguru.py`, it's almost
    # certainly not the third-party dependency.
    return not spec.origin.replace("\\", "/").endswith("/loguru.py")


logger: Any

if _has_real_loguru():
    from loguru import logger as _logger

    logger = _logger
else:
    LEVELS = {
        "DEBUG": 10,
        "INFO": 20,
        "WARNING": 30,
        "ERROR": 40,
    }

    @dataclass
    class _Sink:
        target: TextIO | Path
        level: str
        encoding: str = "utf-8"

    class _FallbackLogger:
        def __init__(self):
            self._sinks: list[_Sink] = [_Sink(sys.stderr, "DEBUG")]

        def remove(self):
            self._sinks.clear()

        def add(self, sink, level="INFO", encoding="utf-8", **kwargs):
            target = Path(sink) if isinstance(sink, (str, Path)) else sink
            self._sinks.append(_Sink(target=target, level=level.upper(), encoding=encoding))
            return len(self._sinks)

        def _should_write(self, sink_level: str, event_level: str) -> bool:
            return LEVELS.get(event_level, 20) >= LEVELS.get(sink_level, 20)

        def _write(self, message: str, level: str):
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"{ts} | {level:<8} | {message}\n"
            for sink in self._sinks:
                if not self._should_write(sink.level, level):
                    continue
                if isinstance(sink.target, Path):
                    sink.target.parent.mkdir(parents=True, exist_ok=True)
                    with open(sink.target, "a", encoding=sink.encoding) as fh:
                        fh.write(line)
                else:
                    sink.target.write(line)
                    sink.target.flush()

        def debug(self, message: str):
            self._write(message, "DEBUG")

        def info(self, message: str):
            self._write(message, "INFO")

        def warning(self, message: str):
            self._write(message, "WARNING")

        def error(self, message: str):
            self._write(message, "ERROR")

        def exception(self, message: str):
            exc = traceback.format_exc()
            if exc.strip() == "NoneType: None":
                self._write(message, "ERROR")
            else:
                self._write(f"{message}\n{exc.rstrip()}", "ERROR")

    logger = _FallbackLogger()
