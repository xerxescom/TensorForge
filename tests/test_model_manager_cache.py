from __future__ import annotations

import os
import time

from tensorforge.model_manager import ModelDownloader


def _write_bytes(path, n: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\0" * n)


def test_enforce_cache_size_limit_evictions(tmp_path):
    d = ModelDownloader(cache_dir=str(tmp_path))
    # Force a tiny cache limit so eviction triggers.
    d.max_cache_size_bytes = 150

    a = tmp_path / "model_a"
    b = tmp_path / "model_b"
    _write_bytes(a / "x.bin", 100)
    _write_bytes(b / "y.bin", 100)

    # Ensure a is older than b.
    now = time.time()
    os.utime(a, (now - 1000, now - 1000))
    os.utime(b, (now - 100, now - 100))

    d._enforce_cache_size_limit()

    # After eviction, total must be <= 150. Since each dir is ~100, one should be gone.
    existing = [p.name for p in tmp_path.iterdir() if p.is_dir()]
    assert len(existing) == 1
    assert existing[0] in {"model_a", "model_b"}
