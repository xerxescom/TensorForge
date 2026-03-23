from pathlib import Path
import sys

<<<<<<<< HEAD:src/tensorforge/core/logging_utils.py
from .tf_logger import logger
========
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
>>>>>>>> 48d5659 (Refactor project into package structure):logging_utils.py

from tensorforge.core.logging_utils import *
