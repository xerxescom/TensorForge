from __future__ import annotations

import subprocess

from tensorforge.run_suite import ConcurrentStressTest


class _Result:
    def __init__(self, stdout: str = ""):
        self.stdout = stdout


def test_timed_llm_records_failures_and_excludes_failed_rounds(monkeypatch):
    bench = ConcurrentStressTest(tasks=[{"type": "llm", "model": "demo"}])

    time_values = iter([100.0, 100.0, 100.2])
    monkeypatch.setattr("tensorforge.run_suite.time.time", lambda: next(time_values))

    def _raise(*args, **kwargs):
        assert kwargs["check"] is True
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=args[0],
            stderr="mock ollama failure",
        )

    monkeypatch.setattr("tensorforge.run_suite.subprocess.run", _raise)

    metrics = bench._timed_llm(model="llama3.1:8b", duration_s=0.1)

    assert metrics["successful_rounds"] == 0
    assert metrics["failed_rounds"] == 1
    assert metrics["failure_rate"] == 1.0
    assert metrics["tokens_generated"] == 0
    assert metrics["error_stage"] == "concurrent_llm_inference_subprocess"


def test_timed_llm_success_rounds_include_check_true(monkeypatch):
    bench = ConcurrentStressTest(tasks=[{"type": "llm", "model": "demo"}])

    time_values = iter([200.0, 200.0, 200.4, 201.2])
    monkeypatch.setattr("tensorforge.run_suite.time.time", lambda: next(time_values))

    calls = []

    def _ok(*args, **kwargs):
        calls.append(kwargs)
        return _Result(stdout="hello world")

    monkeypatch.setattr("tensorforge.run_suite.subprocess.run", _ok)

    metrics = bench._timed_llm(model="llama3.1:8b", duration_s=1.0)

    assert metrics["successful_rounds"] == 2
    assert metrics["failed_rounds"] == 0
    assert metrics["failure_rate"] == 0.0
    assert metrics["tokens_generated"] > 0
    assert all(call["check"] is True for call in calls)
