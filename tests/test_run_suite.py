import json

from run_suite import FullBenchmarkSuite


def test_final_report_contains_config_snapshot(tmp_path):
    suite = FullBenchmarkSuite(output_dir=str(tmp_path), gpu_name="Test GPU", skip_phases=["llm"])
    suite.all_results = [{"phase": "dummy", "result": {"ok": True}}]

    suite._write_final_report()

    report = json.loads((tmp_path / "final_report.json").read_text(encoding="utf-8"))
    assert report["gpu_name"] == "Test GPU"
    assert "config_snapshot" in report
    assert report["config_snapshot"]["llm_model"]
