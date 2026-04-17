from __future__ import annotations

import json
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


def test_enforce_cache_prefers_metadata_and_fallback_scan(tmp_path):
    d = ModelDownloader(cache_dir=str(tmp_path))
    d.max_cache_size_bytes = 150

    old = tmp_path / "old_model"
    new = tmp_path / "new_model"
    _write_bytes(old / "x.bin", 100)
    _write_bytes(new / "y.bin", 100)

    now = time.time()
    metadata = {
        "old_model": {"size_bytes": 100, "last_access_ts": now - 1000},
        "new_model": {"size_bytes": 100, "last_access_ts": now - 10},
    }
    d.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    d._cache_metadata = d._load_cache_metadata()

    d._enforce_cache_size_limit()

    remaining = [p.name for p in tmp_path.iterdir() if p.is_dir()]
    assert remaining == ["new_model"]

    # Remove size metadata so next pass must fallback to realtime scan.
    d._cache_metadata["new_model"].pop("size_bytes", None)
    d._save_cache_metadata()
    d.max_cache_size_bytes = 1000
    d._enforce_cache_size_limit()

    refreshed = json.loads(d.metadata_path.read_text(encoding="utf-8"))
    assert refreshed["new_model"]["size_bytes"] > 0
