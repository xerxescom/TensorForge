"""
Model download and cache management.
"""

from __future__ import annotations

import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from .config_manager import config_manager
from .error_handler import RetryStrategy, retry_on_error
from .network_optimizer import DownloadOptimizer, NetworkConfig
from .tf_logger import logger


@dataclass
class ModelConfig:
    name: str
    huggingface_id: str
    local_path: str | None = None
    cache_dir: str = "models_cache"
    timeout: int = 300
    retry_count: int = 3
    use_mirror: bool = True


class ModelDownloader:
    def __init__(self, cache_dir: str = "models_cache"):
        config = config_manager.load_config()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.download_locks: dict[str, threading.Lock] = {}
        self.mirrors = [*list(config.network_preferred_endpoints), "https://cdn-lfs.huggingface.co"]
        self.network_config = NetworkConfig(
            timeout=config.network_timeout,
            max_retries=config.network_max_retries,
            retry_delay=config.network_retry_delay,
            use_proxy=config.network_use_proxy,
            proxy_host=config.network_proxy_host,
            proxy_port=config.network_proxy_port,
            verify_ssl=config.network_verify_ssl,
            cache_dir=str(self.cache_dir),
            enable_hf_transfer=config.network_enable_hf_transfer,
            preferred_endpoints=list(config.network_preferred_endpoints),
        )
        self.download_optimizer = DownloadOptimizer(self.network_config)
        self.max_cache_size_bytes = int(config.cache_max_size_gb * (1024**3))
        self.cleanup_old_models = config.cache_cleanup_old_models
        self.model_retention_days = config.cache_model_retention_days

    def get_lock(self, model_id: str) -> threading.Lock:
        if model_id not in self.download_locks:
            self.download_locks[model_id] = threading.Lock()
        return self.download_locks[model_id]

    def is_model_cached(self, model_id: str) -> bool:
        cache_path = self.cache_dir / model_id.replace("/", "_")
        return cache_path.exists() and any(cache_path.iterdir())

    def _dir_size_bytes(self, path: Path) -> int:
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())

    def _cleanup_expired_cache(self):
        if not self.cleanup_old_models:
            return
        now = time.time()
        retention_seconds = self.model_retention_days * 24 * 60 * 60
        for model_dir in self.cache_dir.iterdir():
            if not model_dir.is_dir():
                continue
            age_seconds = now - model_dir.stat().st_mtime
            if age_seconds > retention_seconds:
                shutil.rmtree(model_dir, ignore_errors=True)
                logger.info(f"[cache] Removed expired cache directory: {model_dir.name}")

    def _enforce_cache_size_limit(self):
        if self.max_cache_size_bytes <= 0:
            return
        model_dirs = [p for p in self.cache_dir.iterdir() if p.is_dir()]
        total_size = sum(self._dir_size_bytes(p) for p in model_dirs)
        if total_size <= self.max_cache_size_bytes:
            return
        for model_dir in sorted(model_dirs, key=lambda p: p.stat().st_mtime):
            if total_size <= self.max_cache_size_bytes:
                break
            dir_size = self._dir_size_bytes(model_dir)
            shutil.rmtree(model_dir, ignore_errors=True)
            total_size -= dir_size
            logger.info(f"[cache] Evicted {model_dir.name} to enforce cache size limit")

    def download_with_retry(
        self, url: str, local_path: Path, timeout: int = 300, max_retries: int = 3
    ) -> bool:
        strategy = RetryStrategy(max_retries=max_retries, base_delay=1.0, backoff_factor=2.0)

        @retry_on_error(strategy=strategy)
        def _download_once():
            response = requests.get(
                url,
                stream=True,
                timeout=timeout,
                headers={"User-Agent": "TensorForge-Benchmark/1.0"},
            )
            response.raise_for_status()
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return True

        try:
            return bool(_download_once())
        except Exception as e:
            logger.warning(f"[download] Failed after retries: {url} ({type(e).__name__}: {e})")
            return False

    def try_mirrors(self, model_id: str, filename: str, local_path: Path) -> bool:
        base_url = f"https://huggingface.co/{model_id}/resolve/main/{filename}"
        if self.download_with_retry(base_url, local_path):
            return True
        for mirror in self.mirrors[1:]:
            mirror_url = f"{mirror}/{model_id}/resolve/main/{filename}"
            if self.download_with_retry(mirror_url, local_path):
                return True
        return False

    def download_model(self, model_config: ModelConfig) -> str:
        model_id = model_config.huggingface_id
        cache_path = self.cache_dir / model_id.replace("/", "_")
        self._cleanup_expired_cache()
        self._enforce_cache_size_limit()

        if self.is_model_cached(model_id):
            logger.info(f"[cache] Model already cached: {model_id}")
            return str(cache_path)

        lock = self.get_lock(model_id)
        with lock:
            if self.is_model_cached(model_id):
                return str(cache_path)

            logger.info(f"[download] Starting download: {model_id}")
            self.download_optimizer.setup()
            try:
                from huggingface_hub import snapshot_download

                downloaded_path = snapshot_download(
                    repo_id=model_id,
                    cache_dir=str(self.cache_dir),
                    resume_download=True,
                    timeout=model_config.timeout or self.network_config.timeout,
                    max_retries=model_config.retry_count or self.network_config.max_retries,
                )
                self._enforce_cache_size_limit()
                return downloaded_path
            except Exception as e:
                logger.warning(f"[download] Failed with huggingface_hub: {e}")
                return self._manual_download(model_config, cache_path)
            finally:
                self.download_optimizer.cleanup()

    def _manual_download(self, model_config: ModelConfig, cache_path: Path) -> str:
        cache_path.mkdir(parents=True, exist_ok=True)
        essential_files = [
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "unet/config.json",
            "vae/config.json",
        ]
        for filename in essential_files:
            local_file = cache_path / filename
            if not local_file.exists():
                self.try_mirrors(model_config.huggingface_id, filename, local_file)
        return str(cache_path)


class ModelManager:
    def __init__(self, cache_dir: str = "models_cache"):
        config = config_manager.load_config()
        resolved_cache_dir = cache_dir or config.cache_base_dir
        if cache_dir == "models_cache":
            resolved_cache_dir = config.cache_base_dir
        self.downloader = ModelDownloader(resolved_cache_dir)
        self.predefined_models: dict[str, ModelConfig] = {
            "sdxl-turbo": ModelConfig(
                name="SDXL-Turbo",
                huggingface_id="stabilityai/sdxl-turbo",
                timeout=600,
                retry_count=5,
            ),
            "yolov8n": ModelConfig(
                name="YOLOv8n",
                huggingface_id="ultralytics/yolov8n",
                timeout=300,
                retry_count=3,
            ),
            "whisper-base": ModelConfig(
                name="Whisper Base",
                huggingface_id="openai/whisper-base",
                timeout=300,
                retry_count=3,
            ),
        }

    def get_model_path(self, model_name: str) -> str:
        if model_name in self.predefined_models:
            cfg = self.predefined_models[model_name]
            return self.downloader.download_model(cfg)
        cfg = ModelConfig(name=model_name, huggingface_id=model_name)
        return self.downloader.download_model(cfg)

    def add_custom_model(self, name: str, huggingface_id: str, **kwargs):
        self.predefined_models[name] = ModelConfig(
            name=name, huggingface_id=huggingface_id, **kwargs
        )

    def clear_cache(self, model_name: str | None = None):
        if model_name:
            cache_path = self.downloader.cache_dir / model_name.replace("/", "_")
            if cache_path.exists():
                shutil.rmtree(cache_path)
                logger.info(f"[cache] Cleared cache for: {model_name}")
        else:
            if self.downloader.cache_dir.exists():
                shutil.rmtree(self.downloader.cache_dir)
                self.downloader.cache_dir.mkdir(exist_ok=True)
                logger.info("[cache] Cleared all cache")


model_manager = ModelManager()
