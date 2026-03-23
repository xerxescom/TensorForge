from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.tensorforge.reporting.generate_report import generate

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_report.py <final_report.json> [output.html]")
        sys.exit(1)
    generate(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
