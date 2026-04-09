"""
Backward-compatible wrapper for the suite entrypoint.

The implementation lives in `tensorforge.run_suite`.
"""

from __future__ import annotations

from tensorforge.run_suite import main

if __name__ == "__main__":
    raise SystemExit(main())
