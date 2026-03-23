"""
网络连接优化器
解决 Hugging Face 下载超时和网络问题
"""
import os
import socket
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple


@dataclass
class NetworkConfig:
    """网络配置"""
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 2.0
    use_proxy: bool = False
    proxy_host: Optional[str] = None
    proxy_port: Optional[int] = None
    verify_ssl: bool = True
    user_agent: str = "TensorForge-Benchmark/1.0"


class NetworkOptimizer:
    """网络优化器"""
    
    def __init__(self, config: NetworkConfig = None):
        self.config = config or NetworkConfig()
        self.original_env = {}
        
    def setup_environment(self):
        """设置网络环境变量"""
        # 保存原始环境变量
        env_keys = [
            'HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
            'HF_HUB_DISABLE_TELEMETRY', 'HF_HUB_ENABLE_HF_TRANSFER',
            'TRANSFORMERS_CACHE', 'HF_HOME', 'HF_ENDPOINT'
        ]
        
        for key in env_keys:
            self.original_env[key] = os.environ.get(key)
        
        # 优化 Hugging Face 下载
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
        
        # 设置缓存目录
        cache_dir = Path("models_cache")
        cache_dir.mkdir(exist_ok=True)
        os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
        os.environ['HF_HOME'] = str(cache_dir)
        
        # 设置用户代理
        os.environ['HF_USER_AGENT'] = self.config.user_agent
        
        # 代理设置
        if self.config.use_proxy and self.config.proxy_host:
            proxy_url = f"http://{self.config.proxy_host}:{self.config.proxy_port}"
            os.environ['HTTP_PROXY'] = proxy_url
            os.environ['HTTPS_PROXY'] = proxy_url
            print(f"[network] Using proxy: {proxy_url}")
        
        # SSL 验证
        if not self.config.verify_ssl:
            os.environ['PYTHONHTTPSVERIFY'] = '0'
            print("[network] SSL verification disabled")
        
        print("[network] Environment optimized for Hugging Face downloads")
    
    def restore_environment(self):
        """恢复原始环境变量"""
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        print("[network] Environment restored")
    
    def test_connectivity(self, urls: List[str] = None) -> Dict[str, bool]:
        """测试网络连接"""
        if urls is None:
            urls = [
                "https://huggingface.co",
                "https://hf-mirror.com",
                "https://cdn-lfs.huggingface.co"
            ]
        
        results = {}
        print("[network] Testing connectivity...")
        
        for url in urls:
            try:
                start_time = time.time()
                response = urllib.request.urlopen(url, timeout=self.config.timeout)
                elapsed = time.time() - start_time
                results[url] = response.status == 200
                print(f"[network] {url}: OK ({elapsed:.2f}s)")
            except Exception as e:
                results[url] = False
                print(f"[network] {url}: FAILED ({e})")
        
        return results
    
    def get_best_endpoint(self, test_urls: List[str] = None) -> str:
        """获取最佳端点"""
        if test_urls is None:
            test_urls = [
                "https://huggingface.co",
                "https://hf-mirror.com"
            ]
        
        connectivity = self.test_connectivity(test_urls)
        
        # 返回第一个可用的端点
        for url, is_available in connectivity.items():
            if is_available:
                print(f"[network] Selected endpoint: {url}")
                return url
        
        # 如果都不可用，返回默认
        print("[network] No endpoints available, using default")
        return "https://huggingface.co"
    
    def optimize_socket_settings(self):
        """优化 socket 设置"""
        # 设置 socket 超时
        socket.setdefaulttimeout(self.config.timeout)
        
        # 优化缓冲区大小
        try:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context if not self.config.verify_ssl else ssl.create_default_context
        except ImportError:
            pass


class ProxyManager:
    """代理管理器"""
    
    @staticmethod
    def detect_system_proxy() -> Optional[Tuple[str, int]]:
        """检测系统代理设置"""
        import urllib.request
        
        try:
            # 获取系统代理
            proxy_handler = urllib.request.getproxies()
            http_proxy = proxy_handler.get('http')
            https_proxy = proxy_handler.get('https')
            
            if http_proxy:
                # 解析代理地址
                if '://' in http_proxy:
                    http_proxy = http_proxy.split('://', 1)[1]
                
                if ':' in http_proxy:
                    host, port = http_proxy.split(':', 1)
                    return host.strip(), int(port.strip())
            
        except Exception as e:
            print(f"[proxy] System proxy detection failed: {e}")
        
        return None
    
    @staticmethod
    def test_proxy(host: str, port: int, timeout: int = 10) -> bool:
        """测试代理连接"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False


class DownloadOptimizer:
    """下载优化器"""
    
    def __init__(self):
        self.network_config = NetworkConfig()
        self.network_optimizer = NetworkOptimizer(self.network_config)
        
    def setup(self):
        """设置下载优化"""
        # 检测系统代理
        proxy = ProxyManager.detect_system_proxy()
        if proxy:
            host, port = proxy
            if ProxyManager.test_proxy(host, port):
                self.network_config.use_proxy = True
                self.network_config.proxy_host = host
                self.network_config.proxy_port = port
                print(f"[download] Detected system proxy: {host}:{port}")
        
        # 设置环境
        self.network_optimizer.setup_environment()
        self.network_optimizer.optimize_socket_settings()
        
        # 测试连接
        best_endpoint = self.network_optimizer.get_best_endpoint()
        if best_endpoint != "https://huggingface.co":
            print(f"[download] Using mirror endpoint: {best_endpoint}")
    
    def cleanup(self):
        """清理设置"""
        self.network_optimizer.restore_environment()


# 全局下载优化器
download_optimizer = DownloadOptimizer()


def with_download_optimization(func):
    """下载优化装饰器"""
    def wrapper(*args, **kwargs):
        try:
            download_optimizer.setup()
            return func(*args, **kwargs)
        finally:
            download_optimizer.cleanup()
    return wrapper


if __name__ == "__main__":
    # 测试网络连接
    optimizer = NetworkOptimizer()
    connectivity = optimizer.test_connectivity()
    best_endpoint = optimizer.get_best_endpoint()
    
    print(f"\nBest endpoint: {best_endpoint}")
    print("Connectivity results:")
    for url, ok in connectivity.items():
        print(f"  {url}: {'✓' if ok else '✗'}")
