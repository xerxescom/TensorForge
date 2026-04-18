from __future__ import annotations

import textwrap

import pytest

from tensorforge.config_manager import BenchmarkConfig, ConfigManager, apply_overrides


def test_load_config_from_yaml(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            default_settings:
              sample_interval_s: 0.25
              warmup_s: 1.5
              max_raw_samples: 123
              adaptive_sampling: false
              stats_precision_mode: approximate
            network:
              timeout: 10
              max_retries: 2
              retry_delay: 0.5
              use_proxy: false
              verify_ssl: true
              preferred_endpoints:
                - "https://huggingface.co"
            cache:
              base_dir: "x_cache"
              max_size_gb: 1
            models:
              llm:
                default: "llama3.1:8b"
                precision: "q4_k_m"
                backend: "ollama"
                prompts:
                  - "hi"
              cv:
                batch_size: 4
            concurrent_test:
              duration_s: 7
            """
        ).strip(),
        encoding="utf-8",
    )

    cm = ConfigManager(str(cfg_path))
    cfg = cm.load_config()
    assert isinstance(cfg, BenchmarkConfig)
    assert cfg.sample_interval_s == 0.25
    assert cfg.warmup_s == 1.5
    assert cfg.max_raw_samples == 123
    assert cfg.adaptive_sampling is False
    assert cfg.stats_precision_mode == "approximate"
    assert cfg.network_timeout == 10
    assert cfg.cache_base_dir == "x_cache"
    assert cfg.concurrent_duration_s == 7
    assert cfg.llm_prompts == ["hi"]
    assert cfg.cv_batch_size == 4


def test_apply_overrides_type_coercion():
    cfg = BenchmarkConfig()
    apply_overrides(
        cfg,
        {
            "sample_interval_s": "0.1",
            "network_timeout": "600",
            "adaptive_sampling": "false",
            "stats_precision_mode": "approximate",
        },
    )
    assert cfg.sample_interval_s == pytest.approx(0.1)
    assert cfg.network_timeout == 600
    assert cfg.adaptive_sampling is False
    assert cfg.stats_precision_mode == "approximate"


def test_apply_overrides_supports_json_list():
    cfg = BenchmarkConfig()
    apply_overrides(cfg, {"llm_prompts": 'json:["a","b"]'})
    assert cfg.llm_prompts == ["a", "b"]


def test_apply_overrides_rejects_non_json_list():
    cfg = BenchmarkConfig()
    with pytest.raises(ValueError, match="expects a list"):
        apply_overrides(cfg, {"llm_prompts": "a,b"})


def test_apply_overrides_optional_type_validation():
    cfg = BenchmarkConfig()
    apply_overrides(cfg, {"network_proxy_host": "proxy.local"})
    assert cfg.network_proxy_host == "proxy.local"
    apply_overrides(cfg, {"network_proxy_host": "null"})
    assert cfg.network_proxy_host is None


def test_apply_overrides_optional_int_validation_error():
    cfg = BenchmarkConfig()
    with pytest.raises(ValueError, match="network_proxy_port"):
        apply_overrides(cfg, {"network_proxy_port": "not-int"})


def test_apply_overrides_critical_field_validation_error():
    cfg = BenchmarkConfig()
    with pytest.raises(ValueError, match="stats_precision_mode"):
        apply_overrides(cfg, {"stats_precision_mode": "fast"})


def test_apply_overrides_cv_batch_size_validation_error():
    cfg = BenchmarkConfig()
    with pytest.raises(ValueError, match="cv_batch_size"):
        apply_overrides(cfg, {"cv_batch_size": "0"})


def test_load_config_fallback_when_top_level_block_is_not_mapping(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            default_settings: "oops"
            network:
              - bad
              - block
            cache: 123
            models: "wrong-type"
            """
        ).strip(),
        encoding="utf-8",
    )

    cm = ConfigManager(str(cfg_path))
    cfg = cm.load_config()

    assert cfg.sample_interval_s == 0.5
    assert cfg.network_timeout == 300
    assert cfg.cache_base_dir == "models_cache"
    assert cfg.llm_model == "llama3.1:8b"


def test_load_config_fallback_when_models_child_block_is_not_mapping(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            models:
              llm:
                - not
                - mapping
              cv: "bad"
            """
        ).strip(),
        encoding="utf-8",
    )

    cm = ConfigManager(str(cfg_path))
    cfg = cm.load_config()

    assert cfg.llm_model == "llama3.1:8b"
    assert cfg.cv_batch_size == 1
