"""
Configuration manager - loads settings from YAML.
"""

from __future__ import annotations

import json
import types
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml

from .tf_logger import logger


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""

    sample_interval_s: float = 0.5
    warmup_s: float = 5.0
    timeout_s: int = 120
    max_raw_samples: int = 1000
    adaptive_sampling: bool = True

    cache_base_dir: str = "models_cache"
    cache_max_size_gb: float = 50.0
    cache_cleanup_old_models: bool = True
    cache_model_retention_days: int = 30

    network_timeout: int = 300
    network_max_retries: int = 5
    network_retry_delay: float = 2.0
    network_use_proxy: bool = False
    network_proxy_host: str | None = None
    network_proxy_port: int | None = None
    network_verify_ssl: bool = True
    network_enable_hf_transfer: bool = True
    network_preferred_endpoints: list[str] = field(
        default_factory=lambda: [
            "https://huggingface.co",
            "https://hf-mirror.com",
        ]
    )

    error_enable_retries: bool = True
    error_max_retries: int = 3
    error_retry_backoff_factor: float = 2.0
    error_classify_errors: bool = True
    error_report_errors: bool = True

    # GPU thresholds
    high_utilization: float = 80.0
    low_utilization: float = 20.0
    max_temperature: float = 85.0
    max_power_draw: float = 450.0

    # Model config
    llm_prompts: list[str] = field(
        default_factory=lambda: [
            "Explain the difference between a transformer and an RNN in detail.",
            "Write a Python function to compute Fibonacci numbers recursively with memoization.",
            "Describe the water cycle in 300 words.",
            "What are the main causes and consequences of the French Revolution?",
            "Explain how gradient descent works in machine learning.",
        ]
    )
    llm_model: str = "llama3.1:8b"
    llm_precision: str = "q4_k_m"
    llm_backend: str = "ollama"
    llm_fp16_precision: str = "fp16"

    diffusion_n_images: int = 10
    diffusion_n_steps: int = 20
    diffusion_model: str = "sdxl-turbo"
    diffusion_precision: str = "fp16"
    diffusion_local_files_only: bool = False

    cv_n_frames: int = 200
    cv_model: str = "yolov8n"
    cv_precision: str = "fp16"
    cv_image_size: int = 640

    asr_model: str = "base"
    asr_precision: str = "float16"

    concurrent_duration_s: int = 60
    concurrent_measurement_mode: str = "cold_start"


class ConfigManager:
    """Loads config from YAML and maps into BenchmarkConfig."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config: BenchmarkConfig | None = None

    def reset_cache(self):
        self._config = None

    def set_path(self, config_path: str):
        self.config_path = Path(config_path)
        self.reset_cache()

    def load_config(self, *, reload: bool = False) -> BenchmarkConfig:
        if reload:
            self.reset_cache()

        if self._config is None:
            if self.config_path.exists():
                try:
                    data: Any
                    with open(self.config_path, encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                except yaml.YAMLError as e:
                    logger.warning(
                        f"Config file {self.config_path} YAML parse failed: {e}; using defaults"
                    )
                    self._config = BenchmarkConfig()
                    return self._config

                if data is None:
                    logger.warning(f"Config file {self.config_path} is empty, using defaults")
                    self._config = BenchmarkConfig()
                elif not isinstance(data, dict):
                    logger.warning(
                        f"Config file {self.config_path} must be a YAML mapping, got {type(data).__name__}; using defaults"
                    )
                    self._config = BenchmarkConfig()
                else:
                    self._config = self._dict_to_config(data)
            else:
                logger.warning(f"Config file {self.config_path} not found, using defaults")
                self._config = BenchmarkConfig()
        return self._config

    def _dict_to_config(self, data: dict[str, Any]) -> BenchmarkConfig:
        default_settings = data.get("default_settings", {})
        gpu_thresholds = data.get("gpu_thresholds", {})
        models = data.get("models", {})
        network = data.get("network", {})
        cache = data.get("cache", {})
        error_handling = data.get("error_handling", {})
        llm_model_cfg = models.get("llm", {})
        diffusion_model_cfg = models.get("diffusion", {})
        cv_model_cfg = models.get("cv", {})
        asr_model_cfg = models.get("asr", {})
        concurrent_test = data.get("concurrent_test", {})

        llm_prompts = None
        if (
            isinstance(models, dict)
            and isinstance(models.get("llm"), dict)
            and "prompts" in models["llm"]
        ):
            llm_prompts = models["llm"]["prompts"]

        return BenchmarkConfig(
            sample_interval_s=default_settings.get("sample_interval_s", 0.5),
            warmup_s=default_settings.get("warmup_s", 5.0),
            timeout_s=default_settings.get("timeout_s", 120),
            max_raw_samples=default_settings.get("max_raw_samples", 1000),
            adaptive_sampling=default_settings.get("adaptive_sampling", True),
            cache_base_dir=cache.get("base_dir", "models_cache"),
            cache_max_size_gb=cache.get("max_size_gb", 50.0),
            cache_cleanup_old_models=cache.get("cleanup_old_models", True),
            cache_model_retention_days=cache.get("model_retention_days", 30),
            network_timeout=network.get("timeout", 300),
            network_max_retries=network.get("max_retries", 5),
            network_retry_delay=network.get("retry_delay", 2.0),
            network_use_proxy=network.get("use_proxy", False),
            network_proxy_host=network.get("proxy_host"),
            network_proxy_port=network.get("proxy_port"),
            network_verify_ssl=network.get("verify_ssl", True),
            network_enable_hf_transfer=network.get("enable_hf_transfer", True),
            network_preferred_endpoints=network.get(
                "preferred_endpoints",
                ["https://huggingface.co", "https://hf-mirror.com"],
            ),
            error_enable_retries=error_handling.get("enable_retries", True),
            error_max_retries=error_handling.get("max_retries", 3),
            error_retry_backoff_factor=error_handling.get("retry_backoff_factor", 2.0),
            error_classify_errors=error_handling.get("classify_errors", True),
            error_report_errors=error_handling.get("report_errors", True),
            high_utilization=gpu_thresholds.get("high_utilization", 80.0),
            low_utilization=gpu_thresholds.get("low_utilization", 20.0),
            max_temperature=gpu_thresholds.get("max_temperature", 85.0),
            max_power_draw=gpu_thresholds.get("max_power_draw", 450.0),
            llm_prompts=llm_prompts or BenchmarkConfig().llm_prompts,
            llm_model=llm_model_cfg.get("default", "llama3.1:8b"),
            llm_precision=llm_model_cfg.get("precision", "q4_k_m"),
            llm_backend=llm_model_cfg.get("backend", "ollama"),
            llm_fp16_precision=llm_model_cfg.get("fp16_precision", "fp16"),
            diffusion_n_images=diffusion_model_cfg.get("n_images", 10),
            diffusion_n_steps=diffusion_model_cfg.get("n_steps", 20),
            diffusion_model=diffusion_model_cfg.get("default", "sdxl-turbo"),
            diffusion_precision=diffusion_model_cfg.get("precision", "fp16"),
            diffusion_local_files_only=diffusion_model_cfg.get("local_files_only", False),
            cv_n_frames=cv_model_cfg.get("n_frames", 200),
            cv_model=cv_model_cfg.get("default", "yolov8n"),
            cv_precision=cv_model_cfg.get("precision", "fp16"),
            cv_image_size=cv_model_cfg.get("image_size", 640),
            asr_model=asr_model_cfg.get("default", "base"),
            asr_precision=asr_model_cfg.get("precision", "float16"),
            concurrent_duration_s=concurrent_test.get("duration_s", 60),
            concurrent_measurement_mode=concurrent_test.get("measurement_mode", "cold_start"),
        )


def _parse_bool(s: str) -> bool:
    v = s.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean: {s!r}")


def _coerce_value(target_type: type, raw: str):
    if target_type is bool:
        return _parse_bool(raw)
    if target_type is int:
        return int(raw)
    if target_type is float:
        return float(raw)
    return raw


def _is_optional_type(tp: Any) -> bool:
    origin = get_origin(tp)
    if origin in (Union, types.UnionType):
        return type(None) in get_args(tp)
    return False


def _parse_typed_value(key: str, target_type: Any, raw: str) -> Any:
    origin = get_origin(target_type)
    args = get_args(target_type)

    if _is_optional_type(target_type):
        non_none = [t for t in args if t is not type(None)]
        if raw.strip().lower() in {"none", "null"}:
            return None
        if len(non_none) == 1:
            return _parse_typed_value(key, non_none[0], raw)

    if raw.startswith("json:"):
        try:
            parsed = json.loads(raw[len("json:") :])
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON override for {key}: {e}") from e
    else:
        parsed = raw

    if target_type is bool:
        if not isinstance(parsed, str):
            raise ValueError(f"{key} expects bool string, got {type(parsed).__name__}")
        return _parse_bool(parsed)
    if target_type is int:
        if not isinstance(parsed, str):
            raise ValueError(f"{key} expects int string, got {type(parsed).__name__}")
        return int(parsed)
    if target_type is float:
        if not isinstance(parsed, str):
            raise ValueError(f"{key} expects float string, got {type(parsed).__name__}")
        return float(parsed)
    if target_type is str:
        if not isinstance(parsed, str):
            raise ValueError(f"{key} expects string, got {type(parsed).__name__}")
        return parsed

    if origin is list:
        if isinstance(parsed, str):
            raise ValueError(
                f"{key} expects a list. Use json: prefix, e.g. --set {key}=json:[\"a\",\"b\"]"
            )
        if not isinstance(parsed, list):
            raise ValueError(f"{key} expects list, got {type(parsed).__name__}")
        elem_type = args[0] if args else Any
        if elem_type in (str, int, float, bool):
            for idx, item in enumerate(parsed):
                if not isinstance(item, elem_type):
                    raise ValueError(
                        f"{key}[{idx}] expects {elem_type.__name__}, got {type(item).__name__}"
                    )
        return parsed

    if origin is dict:
        if isinstance(parsed, str):
            raise ValueError(
                f"{key} expects a dict. Use json: prefix, e.g. --set {key}=json:{{\"k\":\"v\"}}"
            )
        if not isinstance(parsed, dict):
            raise ValueError(f"{key} expects dict, got {type(parsed).__name__}")
        return parsed

    # Fallback for unsupported annotation shapes.
    return _coerce_value(type(parsed), parsed) if isinstance(parsed, str) else parsed


def apply_overrides(cfg: BenchmarkConfig, overrides: dict[str, str]) -> BenchmarkConfig:
    """
    Apply CLI overrides to an existing config.

    Keys are dataclass field names, e.g. `sample_interval_s=0.25` or `network_timeout=600`.
    """
    type_hints = get_type_hints(BenchmarkConfig)
    allowed = {f.name: type_hints.get(f.name, f.type) for f in fields(BenchmarkConfig)}
    for key, raw in overrides.items():
        if key not in allowed:
            raise KeyError(f"Unknown config key: {key}")
        target_type = allowed[key]
        try:
            value = _parse_typed_value(key, target_type, raw)
        except Exception as e:
            raise ValueError(f"Invalid override for {key!r}: {e}") from e
        setattr(cfg, key, value)

    return cfg


config_manager = ConfigManager()
