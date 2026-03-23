from pathlib import Path
import sys

<<<<<<<< HEAD:src/tensorforge/core/error_handler.py
from .config_manager import config_manager
from .tf_logger import logger
========
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
>>>>>>>> 48d5659 (Refactor project into package structure):error_handler.py

from tensorforge.core.error_handler import *
