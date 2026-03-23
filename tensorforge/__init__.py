"""Compatibility package for local development.

Allows imports like `from tensorforge.benchmarks.llm import LLMBenchmark`
when working directly from the repository root without installing the
package first.
"""
from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent
_src_pkg_dir = _pkg_dir.parent / "src" / "tensorforge"

__path__ = [str(_src_pkg_dir)] if _src_pkg_dir.exists() else []
