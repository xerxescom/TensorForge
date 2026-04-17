from __future__ import annotations

import json

from tensorforge.run_suite import FullBenchmarkSuite


def test_final_report_schema_minimal(tmp_path):
    suite = FullBenchmarkSuite(
        output_dir=str(tmp_path),
        gpu_name="TEST_GPU",
        gpu_index=0,
        skip_phases=[
            "llm",
            "llm_fp16",
            "llm_context_scale",
            "diffusion",
            "cv",
            "asr",
            "concurrent",
        ],
        config_path="config.yaml",
        config_overrides={},
        verbose=False,
    )
    suite.run_all()

    report_path = tmp_path / "final_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["gpu_name"] == "TEST_GPU"
    assert "benchmark_date" in report
    assert "config_snapshot" in report
    assert "phases" in report
    assert isinstance(report["phases"], list)
    assert "error_summary" in report
    assert report["error_summary"] == {
        "download_failed": 0,
        "oom": 0,
        "dependency_missing": 0,
        "timeout": 0,
    }

    cfg_path = tmp_path / "config_resolved.json"
    assert cfg_path.exists()
