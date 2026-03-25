"""
模型管理器 - 处理模型下载、缓存和验证
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List

import requests

from .error_handler import error_handler


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    source: str  # huggingface, local, ollama
    url: Optional[str] = None
    local_path: Optional[str] = None
    size_bytes: Optional[int] = None
    checksum: Optional[str] = None
    format: str = "safetensors"  # safetensors, pytorch, gguf


class ModelManager:
    """模型管理器"""
    
    def __init__(self, cache_dir: str = "models_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_registry: Dict[str, ModelConfig] = {}
        self._load_registry()
    
    def _load_registry(self):
        """加载模型注册表"""
        registry_file = self.cache_dir / "registry.json"
        if registry_file.exists():
            with open(registry_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for name, config_data in data.items():
                    self.model_registry[name] = ModelConfig(**config_data)
    
    def _save_registry(self):
        """保存模型注册表"""
        registry_file = self.cache_dir / "registry.json"
        with open(registry_file, 'w', encoding='utf-8') as f:
            data = {name: {
                'name': config.name,
                'source': config.source,
                'url': config.url,
                'local_path': config.local_path,
                'size_bytes': config.size_bytes,
                'checksum': config.checksum,
                'format': config.format
            } for name, config in self.model_registry.items()}
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def register_model(self, config: ModelConfig):
        """注册模型"""
        self.model_registry[config.name] = config
        self._save_registry()
    
    def get_model_path(self, model_name: str) -> Optional[Path]:
        """获取模型路径"""
        if model_name not in self.model_registry:
            return None
        
        config = self.model_registry[model_name]
        
        if config.source == "local" and config.local_path:
            return Path(config.local_path)
        elif config.source in ["huggingface", "url"]:
            # 检查缓存
            cached_path = self.cache_dir / self._get_cache_filename(model_name)
            if cached_path.exists():
                return cached_path
            else:
                return None
        elif config.source == "ollama":
            # Ollama 模型不需要本地文件
            return None
        
        return None
    
    def _get_cache_filename(self, model_name: str) -> str:
        """生成缓存文件名"""
        # 使用模型名的哈希作为文件名，避免特殊字符问题
        return f"{hashlib.md5(model_name.encode()).hexdigest()}.bin"
    
    def download_model(self, model_name: str, progress_callback=None) -> bool:
        """下载模型"""
        if model_name not in self.model_registry:
            print(f"[model] Unknown model: {model_name}")
            return False
        
        config = self.model_registry[model_name]
        
        if config.source == "local":
            print(f"[model] Using local model: {config.local_path}")
            return True
        
        if not config.url:
            print(f"[model] No download URL for model: {model_name}")
            return False
        
        cached_path = self.cache_dir / self._get_cache_filename(model_name)
        
        # 检查是否已存在且完整
        if cached_path.exists():
            if self._verify_model(cached_path, config):
                print(f"[model] Model already cached: {model_name}")
                return True
            else:
                print(f"[model] Cached model corrupted, re-downloading: {model_name}")
                cached_path.unlink()
        
        # 下载模型
        try:
            return self._download_with_retry(config.url, cached_path, config, progress_callback)
        except Exception as e:
            print(f"[model] Download failed: {e}")
            return False
    
    def _download_with_retry(self, url: str, dest_path: Path, config: ModelConfig, progress_callback=None) -> bool:
        """带重试的下载"""
        def _download():
            return self._single_download(url, dest_path, config, progress_callback)
        
        try:
            error_handler.retry(_download)
            return True
        except Exception as e:
            print(f"[model] Download failed after retries: {e}")
            return False
    
    def _single_download(self, url: str, dest_path: Path, config: ModelConfig, progress_callback=None):
        """单次下载"""
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if progress_callback and total_size > 0:
                        progress_callback(downloaded / total_size)
        
        # 验证下载的文件
        if not self._verify_model(dest_path, config):
            dest_path.unlink()
            raise ValueError("Downloaded file verification failed")
    
    def _verify_model(self, file_path: Path, config: ModelConfig) -> bool:
        """验证模型文件"""
        if not file_path.exists():
            return False
        
        # 检查文件大小
        if config.size_bytes:
            actual_size = file_path.stat().st_size
            if actual_size != config.size_bytes:
                print(f"[model] Size mismatch: expected {config.size_bytes}, got {actual_size}")
                return False
        
        # 检查校验和
        if config.checksum:
            actual_checksum = self._calculate_checksum(file_path)
            if actual_checksum != config.checksum:
                print(f"[model] Checksum mismatch: expected {config.checksum}, got {actual_checksum}")
                return False
        
        return True
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """计算文件校验和"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def list_cached_models(self) -> List[str]:
        """列出已缓存的模型"""
        cached = []
        for model_name, config in self.model_registry.items():
            if self.get_model_path(model_name) and self.get_model_path(model_name).exists():
                cached.append(model_name)
        return cached
    
    def clear_cache(self, model_name: Optional[str] = None):
        """清理缓存"""
        if model_name:
            # 清理特定模型
            if model_name in self.model_registry:
                cached_path = self.get_model_path(model_name)
                if cached_path and cached_path.exists():
                    cached_path.unlink()
                    print(f"[model] Cleared cache for: {model_name}")
        else:
            # 清理所有缓存
            for file_path in self.cache_dir.glob("*.bin"):
                file_path.unlink()
            print("[model] Cleared all model cache")


# 全局模型管理器实例
model_manager = ModelManager()
