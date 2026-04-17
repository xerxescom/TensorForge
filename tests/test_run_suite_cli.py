from __future__ import annotations

import json

from tensorforge import run_suite


class _DummySuite:
    last_kwargs = None

    def __init__(self, **kwargs):
        _DummySuite.last_kwargs = kwargs

    def run_all(self):
        return None


def test_main_supports_set_json_prefix(monkeypatch):
    monkeypatch.setattr(run_suite, "FullBenchmarkSuite", _DummySuite)
    rc = run_suite.main(
        [
            "--config",
            "config.yaml",
            "--set",
            'llm_prompts=json:["a","b"]',
            "--only",
            "llm",
        ]
    )
    assert rc == 0
    assert _DummySuite.last_kwargs["config_overrides"]["llm_prompts"].startswith("json:")


def test_main_supports_set_file_object_injection(tmp_path, monkeypatch):
    monkeypatch.setattr(run_suite, "FullBenchmarkSuite", _DummySuite)
    payload = {"llm_prompts": ["x", "y"], "stats_precision_mode": "approximate"}
    p = tmp_path / "overrides.json"
    p.write_text(json.dumps(payload), encoding="utf-8")

    rc = run_suite.main(["--config", "config.yaml", "--set-file", str(p), "--only", "llm"])
    assert rc == 0
    overrides = _DummySuite.last_kwargs["config_overrides"]
    assert overrides["llm_prompts"].startswith("json:")
    assert overrides["stats_precision_mode"] == "approximate"


def test_main_set_file_bad_type_without_key(tmp_path, monkeypatch):
    monkeypatch.setattr(run_suite, "FullBenchmarkSuite", _DummySuite)
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(["not", "object"]), encoding="utf-8")

    try:
        run_suite.main(["--config", "config.yaml", "--set-file", str(p), "--only", "llm"])
    except SystemExit as e:
        assert "expected JSON/YAML object" in str(e)
    else:
        raise AssertionError("Expected SystemExit for invalid --set-file payload")
