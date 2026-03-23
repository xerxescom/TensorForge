from pathlib import Path
import sys
<<<<<<<< HEAD:src/tensorforge/benchmarks/multimodal.py
import json
from pathlib import Path
from ..core.tf_logger import logger
from ..core.collector import BenchmarkRunner, _subprocess_kwargs
from ..core.model_manager import model_manager
========
>>>>>>>> 48d5659 (Refactor project into package structure):other_bench.py

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensorforge.benchmarks.multimodal import *
