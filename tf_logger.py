"""
Backward-compatible wrapper.

The implementation lives in `tensorforge.tf_logger`.
"""

# NOTE:
# Root-level wrappers are intentionally kept for users that run legacy commands
# like `python tf_logger.py` or import from project root. New code should import
# from `tensorforge.tf_logger` directly.
from tensorforge.tf_logger import *  # noqa: F403
