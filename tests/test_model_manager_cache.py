from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

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


def test_update_model_metadata_supports_source_and_increment(tmp_path):
    d = ModelDownloader(cache_dir=str(tmp_path))
    key = "demo_model"
    d._update_model_metadata(key, size_delta=100, source="mirror_a")
    d._update_model_metadata(key, size_delta=50, source="mirror_b")
    d._save_cache_metadata()

    payload = json.loads(d.metadata_path.read_text(encoding="utf-8"))
    assert payload[key]["size_bytes"] == 150
    assert payload[key]["source"] == "mirror_b"
    assert payload[key]["last_access_ts"] > 0


def test_full_scan_fallback_rebuilds_metadata(tmp_path):
    d = ModelDownloader(cache_dir=str(tmp_path))
    _write_bytes(tmp_path / "model_x" / "file.bin", 64)
    _write_bytes(tmp_path / "model_y" / "file.bin", 32)

    d._fallback_full_scan_rebuild_metadata()
    payload = json.loads(d.metadata_path.read_text(encoding="utf-8"))
    assert payload["model_x"]["size_bytes"] >= 64
    assert payload["model_y"]["size_bytes"] >= 32
    assert payload["model_x"]["source"] == "scan_fallback"


def test_get_lock_returns_single_instance_under_concurrency(tmp_path):
    d = ModelDownloader(cache_dir=str(tmp_path))
    model_id = "org/repo"
    n_workers = 32
    barrier = threading.Barrier(n_workers)

    def _get():
        barrier.wait()
        return d.get_lock(model_id)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        locks = list(pool.map(lambda _: _get(), range(n_workers)))

    first = locks[0]
    assert all(lock is first for lock in locks)


def test_is_model_cached_uses_batched_metadata_flush(tmp_path):
    d = ModelDownloader(cache_dir=str(tmp_path))
    d._metadata_flush_interval_sec = 3600
    d._metadata_flush_batch_updates = 50

    model_dir = tmp_path / "org_repo"
    _write_bytes(model_dir / "weights.bin", 8)

    assert d.is_model_cached("org/repo") is True
    assert d.metadata_path.exists() is False

    d.flush_cache_metadata()
    payload = json.loads(d.metadata_path.read_text(encoding="utf-8"))
    assert payload["org_repo"]["last_access_ts"] > 0
