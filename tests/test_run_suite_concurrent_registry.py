from __future__ import annotations

import threading

from tensorforge.run_suite import ConcurrentStressTest


def test_run_baselines_reuses_registered_executor_interface(monkeypatch):
    bench = ConcurrentStressTest(
        tasks=[
            {"type": "llm", "model": "llama", "duration_s": 11},
            {"type": "diffusion", "model": "sdxl", "duration_s": 7},
        ]
    )

    calls: list[tuple[str, str, float]] = []

    def _fake_llm(model: str, duration_s: float):
        calls.append(("llm", model, duration_s))
        return {"tokens_per_s": 12.5}

    def _fake_diffusion(model: str, duration_s: float):
        calls.append(("diffusion", model, duration_s))
        return {"it_per_s": 3.2}

    bench._task_executors = {"llm": _fake_llm, "diffusion": _fake_diffusion}

    baselines = bench._run_baselines()

    assert baselines["llm_llama"]["tokens_per_s"] == 12.5
    assert baselines["diffusion_sdxl"]["it_per_s"] == 3.2
    assert calls == [("llm", "llama", 11), ("diffusion", "sdxl", 7)]


def test_run_one_task_unsupported_type_uses_structured_error():
    bench = ConcurrentStressTest(tasks=[{"type": "cv", "model": "resnet50"}])
    results = [None]
    barrier = threading.Barrier(1)

    bench._run_one_task(bench.tasks[0], idx=0, results=results, barrier=barrier)

    assert results[0]["type"] == "cv"
    assert results[0]["model"] == "resnet50"
    assert results[0]["error_stage"] == "concurrent_cv_subprocess"
    assert "unsupported concurrent task type: cv" in results[0]["short_trace"]
