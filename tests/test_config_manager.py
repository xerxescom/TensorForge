from config_manager import ConfigManager


def test_config_manager_parses_network_cache_and_error_sections(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
network:
  timeout: 123
  max_retries: 7
  retry_delay: 1.5
  use_proxy: true
  proxy_host: proxy.local
  proxy_port: 8080
  verify_ssl: false
  enable_hf_transfer: false
  preferred_endpoints:
    - https://primary.example
cache:
  base_dir: custom_cache
  max_size_gb: 12
  cleanup_old_models: false
  model_retention_days: 9
error_handling:
  enable_retries: false
  max_retries: 4
  retry_backoff_factor: 3.0
  classify_errors: false
  report_errors: false
""",
        encoding="utf-8",
    )

    cfg = ConfigManager(str(config_path)).load_config()

    assert cfg.network_timeout == 123
    assert cfg.network_max_retries == 7
    assert cfg.network_retry_delay == 1.5
    assert cfg.network_use_proxy is True
    assert cfg.network_proxy_host == "proxy.local"
    assert cfg.network_proxy_port == 8080
    assert cfg.network_verify_ssl is False
    assert cfg.network_enable_hf_transfer is False
    assert cfg.network_preferred_endpoints == ["https://primary.example"]
    assert cfg.cache_base_dir == "custom_cache"
    assert cfg.cache_max_size_gb == 12
    assert cfg.cache_cleanup_old_models is False
    assert cfg.cache_model_retention_days == 9
    assert cfg.error_enable_retries is False
    assert cfg.error_max_retries == 4
    assert cfg.error_retry_backoff_factor == 3.0
    assert cfg.error_classify_errors is False
    assert cfg.error_report_errors is False
