"""
Network connection optimizer.
"""

from __future__ import annotations

import os
import socket
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .tf_logger import logger


@dataclass
class NetworkConfig:
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 2.0
    use_proxy: bool = False
    proxy_host: str | None = None
    proxy_port: int | None = None
    verify_ssl: bool = True
    user_agent: str = "TensorForge-Benchmark/1.0"
    cache_dir: str = "models_cache"
    enable_hf_transfer: bool = True
    preferred_endpoints: list[str] | None = None


class NetworkOptimizer:
    def __init__(self, config: NetworkConfig | None = None):
        self.config = config or NetworkConfig()
        self.original_env: dict[str, str | None] = {}
        if self.config.preferred_endpoints is None:
            self.config.preferred_endpoints = [
                "https://huggingface.co",
                "https://hf-mirror.com",
                "https://cdn-lfs.huggingface.co",
            ]

    def setup_environment(self):
        env_keys = [
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "http_proxy",
            "https_proxy",
            "HF_HUB_DISABLE_TELEMETRY",
            "HF_HUB_ENABLE_HF_TRANSFER",
            "TRANSFORMERS_CACHE",
            "HF_HOME",
            "HF_ENDPOINT",
            "HF_USER_AGENT",
        ]
        for key in env_keys:
            self.original_env[key] = os.environ.get(key)

        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" if self.config.enable_hf_transfer else "0"

        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(exist_ok=True)
        os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
        os.environ["HF_HOME"] = str(cache_dir)
        os.environ["HF_USER_AGENT"] = self.config.user_agent

        if self.config.use_proxy and self.config.proxy_host:
            proxy_url = f"http://{self.config.proxy_host}:{self.config.proxy_port}"
            os.environ["HTTP_PROXY"] = proxy_url
            os.environ["HTTPS_PROXY"] = proxy_url
            logger.info(f"[network] Using proxy: {proxy_url}")

        if not self.config.verify_ssl:
            os.environ["PYTHONHTTPSVERIFY"] = "0"
            logger.warning("[network] SSL verification disabled")

        logger.info("[network] Environment optimized for Hugging Face downloads")

    def restore_environment(self):
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        logger.info("[network] Environment restored")

    def test_connectivity(self, urls: list[str] | None = None) -> dict[str, bool]:
        urls = urls or (self.config.preferred_endpoints or [])
        results: dict[str, bool] = {}
        logger.info(f"[network] Testing connectivity to {len(urls)} endpoints...")

        for i, url in enumerate(urls, 1):
            logger.debug(f"[network] Testing endpoint {i}/{len(urls)}: {url}")
            start_time = time.time()
            try:
                response = urllib.request.urlopen(url, timeout=self.config.timeout)
                elapsed = time.time() - start_time
                is_ok = getattr(response, "status", 200) == 200
                results[url] = is_ok
                if is_ok:
                    logger.info(
                        f"[network] {url}: OK ({elapsed:.2f}s, status={getattr(response, 'status', 200)})"
                    )
                else:
                    logger.warning(
                        f"[network] {url}: HTTP {getattr(response, 'status', 0)} ({elapsed:.2f}s)"
                    )
            except urllib.error.HTTPError as e:
                elapsed = time.time() - start_time
                results[url] = False
                logger.warning(f"[network] {url}: HTTP {e.code} ({elapsed:.2f}s) - {e.reason}")
            except urllib.error.URLError as e:
                elapsed = time.time() - start_time
                results[url] = False
                logger.warning(f"[network] {url}: URL Error ({elapsed:.2f}s) - {e.reason}")
            except TimeoutError as e:
                elapsed = time.time() - start_time
                results[url] = False
                logger.warning(f"[network] {url}: Timeout ({elapsed:.2f}s) - {e}")
            except Exception as e:
                elapsed = time.time() - start_time
                results[url] = False
                logger.warning(
                    f"[network] {url}: FAILED ({elapsed:.2f}s) - {type(e).__name__}: {e}"
                )

        success_count = sum(1 for ok in results.values() if ok)
        logger.info(
            f"[network] Connectivity test complete: {success_count}/{len(urls)} endpoints reachable"
        )
        return results

    def get_best_endpoint(self, test_urls: list[str] | None = None) -> str:
        test_urls = test_urls or (self.config.preferred_endpoints or [])[:2]
        connectivity = self.test_connectivity(test_urls)
        for url, ok in connectivity.items():
            if ok:
                logger.info(f"[network] Selected endpoint: {url}")
                return url
        logger.warning("[network] No endpoints available, using default")
        return "https://huggingface.co"

    def optimize_socket_settings(self):
        socket.setdefaulttimeout(self.config.timeout)
        try:
            import ssl

            if not self.config.verify_ssl:
                ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[assignment]
                logger.warning("[network] SSL verification disabled for better compatibility")
            else:
                ssl._create_default_https_context = ssl.create_default_context
        except ImportError:
            return


class ProxyManager:
    @staticmethod
    def detect_system_proxy() -> tuple[str, int] | None:
        try:
            proxy_handler = urllib.request.getproxies()
            http_proxy = proxy_handler.get("http")
            if http_proxy:
                if "://" in http_proxy:
                    http_proxy = http_proxy.split("://", 1)[1]
                if ":" in http_proxy:
                    host, port = http_proxy.split(":", 1)
                    return host.strip(), int(port.strip())
        except Exception as e:
            logger.warning(f"[proxy] System proxy detection failed: {e}")
        return None

    @staticmethod
    def test_proxy(host: str, port: int, timeout: int = 10) -> bool:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False


class DownloadOptimizer:
    def __init__(self, network_config: NetworkConfig | None = None):
        self.network_config = network_config or NetworkConfig()
        self.network_optimizer = NetworkOptimizer(self.network_config)

    def setup(self):
        proxy = ProxyManager.detect_system_proxy()
        if proxy:
            host, port = proxy
            if ProxyManager.test_proxy(host, port):
                self.network_config.use_proxy = True
                self.network_config.proxy_host = host
                self.network_config.proxy_port = port
                logger.info(f"[download] Detected system proxy: {host}:{port}")

        self.network_optimizer.setup_environment()
        self.network_optimizer.optimize_socket_settings()

        best_endpoint = self.network_optimizer.get_best_endpoint()
        if best_endpoint != "https://huggingface.co":
            logger.info(f"[download] Using mirror endpoint: {best_endpoint}")

    def cleanup(self):
        self.network_optimizer.restore_environment()


download_optimizer = DownloadOptimizer()


def with_download_optimization(func):
    def wrapper(*args, **kwargs):
        try:
            download_optimizer.setup()
            return func(*args, **kwargs)
        finally:
            download_optimizer.cleanup()

    return wrapper
