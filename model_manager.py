"""
模型下载和缓存管理器
解决 Hugging Face 下载超时问题
"""
import os
import time
import shutil
import requests
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import threading
from urllib.parse import urlparse

from tf_logger import logger


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    huggingface_id: str
    local_path: Optional[str] = None
    cache_dir: str = "models_cache"
    timeout: int = 300  # 5分钟
    retry_count: int = 3
    use_mirror: bool = True


class ModelDownloader:
    """模型下载器"""
    
    def __init__(self, cache_dir: str = "models_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.download_locks: Dict[str, threading.Lock] = {}
        self.mirrors = [
            "https://huggingface.co",
            "https://hf-mirror.com",  # 中国镜像
            "https://cdn-lfs.huggingface.co"
        ]
    
    def get_lock(self, model_id: str) -> threading.Lock:
        """获取模型下载锁"""
        if model_id not in self.download_locks:
            self.download_locks[model_id] = threading.Lock()
        return self.download_locks[model_id]
    
    def is_model_cached(self, model_id: str) -> bool:
        """检查模型是否已缓存"""
        cache_path = self.cache_dir / model_id.replace("/", "_")
        return cache_path.exists() and any(cache_path.iterdir())
    
    def download_with_retry(self, url: str, local_path: Path, timeout: int = 300, max_retries: int = 3) -> bool:
        """带重试的文件下载"""
        for attempt in range(max_retries):
            try:
                logger.info(f"[download] Attempt {attempt + 1}/{max_retries}: {url}")
                
                # 使用流式下载，支持大文件
                response = requests.get(
                    url, 
                    stream=True, 
                    timeout=timeout,
                    headers={'User-Agent': 'TensorForge-Benchmark/1.0'}
                )
                response.raise_for_status()
                
                # 创建目录
                local_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 下载文件
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # 显示进度
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                logger.debug(
                                    f"[download] Progress for {local_path.name}: {progress:.1f}%"
                                )
                
                logger.info(f"[download] Success: {local_path}")
                return True
                
            except requests.exceptions.Timeout:
                logger.warning(f"[download] Timeout on attempt {attempt + 1}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # 指数退避
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"[download] Error: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        
        return False
    
    def try_mirrors(self, model_id: str, filename: str, local_path: Path) -> bool:
        """尝试从不同镜像下载"""
        base_url = f"https://huggingface.co/{model_id}/resolve/main/{filename}"
        
        # 尝试主站
        if self.download_with_retry(base_url, local_path):
            return True
        
        # 尝试镜像站
        if self.mirrors:
            for mirror in self.mirrors[1:]:
                mirror_url = f"{mirror}/{model_id}/resolve/main/{filename}"
                logger.info(f"[download] Trying mirror: {mirror}")
                if self.download_with_retry(mirror_url, local_path):
                    return True
        
        return False
    
    def download_model(self, model_config: ModelConfig) -> str:
        """下载完整模型"""
        model_id = model_config.huggingface_id
        cache_path = self.cache_dir / model_id.replace("/", "_")
        
        # 检查是否已缓存
        if self.is_model_cached(model_id):
            logger.info(f"[cache] Model already cached: {model_id}")
            return str(cache_path)
        
        # 使用锁防止并发下载
        lock = self.get_lock(model_id)
        with lock:
            # 双重检查
            if self.is_model_cached(model_id):
                return str(cache_path)
            
            logger.info(f"[download] Starting download: {model_id}")
            
            # 获取模型文件列表
            try:
                from huggingface_hub import hf_hub_download, snapshot_download
                from huggingface_hub.utils import RepositoryNotFoundError
                
                # 使用 snapshot_download 下载整个模型
                downloaded_path = snapshot_download(
                    repo_id=model_id,
                    cache_dir=str(self.cache_dir),
                    resume_download=True,
                    timeout=model_config.timeout,
                    max_retries=model_config.retry_count
                )
                
                logger.info(f"[download] Model downloaded to: {downloaded_path}")
                return downloaded_path
                
            except ImportError:
                # 如果没有 huggingface_hub，使用手动下载
                return self._manual_download(model_config, cache_path)
            except Exception as e:
                logger.warning(f"[download] Failed with huggingface_hub: {e}")
                return self._manual_download(model_config, cache_path)
    
    def _manual_download(self, model_config: ModelConfig, cache_path: Path) -> str:
        """手动下载模型（备用方案）"""
        logger.warning("[download] Using manual download method")
        
        # 创建模型目录
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # 下载关键文件
        essential_files = [
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "unet/config.json",
            "vae/config.json"
        ]
        
        for filename in essential_files:
            local_file = cache_path / filename
            if not local_file.exists():
                if not self.try_mirrors(model_config.huggingface_id, filename, local_file):
                    logger.warning(f"[download] Failed to download: {filename}")
        
        return str(cache_path)


class ModelManager:
    """模型管理器"""
    
    def __init__(self, cache_dir: str = "models_cache"):
        self.downloader = ModelDownloader(cache_dir)
        self.predefined_models = {
            "sdxl-turbo": ModelConfig(
                name="SDXL-Turbo",
                huggingface_id="stabilityai/sdxl-turbo",
                timeout=600,  # 10分钟
                retry_count=5
            ),
            "yolov8n": ModelConfig(
                name="YOLOv8n",
                huggingface_id="ultralytics/yolov8n",
                timeout=300,
                retry_count=3
            ),
            "whisper-base": ModelConfig(
                name="Whisper Base",
                huggingface_id="openai/whisper-base",
                timeout=300,
                retry_count=3
            )
        }
    
    def get_model_path(self, model_name: str) -> str:
        """获取模型路径（自动下载）"""
        if model_name in self.predefined_models:
            config = self.predefined_models[model_name]
            return self.downloader.download_model(config)
        else:
            # 直接使用 huggingface_id
            config = ModelConfig(name=model_name, huggingface_id=model_name)
            return self.downloader.download_model(config)
    
    def add_custom_model(self, name: str, huggingface_id: str, **kwargs):
        """添加自定义模型"""
        self.predefined_models[name] = ModelConfig(
            name=name,
            huggingface_id=huggingface_id,
            **kwargs
        )
    
    def clear_cache(self, model_name: Optional[str] = None):
        """清理缓存"""
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


# 全局模型管理器
model_manager = ModelManager()
