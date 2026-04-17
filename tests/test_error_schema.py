from __future__ import annotations

from tensorforge.error_schema import classify_error_type, ensure_structured_error, structured_error
from tensorforge.run_suite import FullBenchmarkSuite


def test_classify_error_type_known_categories():
    assert classify_error_type("CUDA out of memory while running pipeline") == "oom"
    assert classify_error_type("No module named diffusers") == "dependency_missing"
    assert classify_error_type("request timed out after 120s") == "timeout"
    assert classify_error_type("model download failed: network unreachable") == "download_failed"


def test_ensure_structured_error_adds_required_fields():
    payload = {"error": "process_failed", "details": "Traceback: No module named torch"}
    out = ensure_structured_error(payload, error_stage="cv_worker_subprocess")
    assert out["error_type"] == "dependency_missing"
    assert out["error_stage"] == "cv_worker_subprocess"
    assert out["short_trace"]


def test_collect_error_counts_aggregates_nested_payload():
    counters = {"download_failed": 0, "oom": 0, "dependency_missing": 0, "timeout": 0}
    payload = [
        {"phase": "x", "error": "timeout", "error_type": "timeout"},
        {"result": {"metrics": [structured_error(error="oom", error_type="oom", error_stage="s")]}}
    ]
    FullBenchmarkSuite._collect_error_counts(payload, counters)
    assert counters == {"download_failed": 0, "oom": 1, "dependency_missing": 0, "timeout": 1}

