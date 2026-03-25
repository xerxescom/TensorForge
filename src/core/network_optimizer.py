"""
网络优化器 - 处理网络连接、代理和下载优化
"""
import os
import time
import socket
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from urllib.parse import urlparse
import requests


@dataclass
class NetworkConfig:
    """网络配置"""
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    use_proxy: bool = False
    proxy_host: Optional[str] = None
    proxy_port: Optional[int] = None
    proxy_username: Optional[str] = None
    proxy_password: Optional[str] = None
    user_agent: str = "TensorForge/1.0"
    verify_ssl: bool = True


class NetworkOptimizer:
    """网络优化器"""
    
    def __init__(self, config: Optional[NetworkConfig] = None):
        self.config = config or self._detect_config()
        self.session = self._create_session()
    
    def _detect_config(self) -> NetworkConfig:
        """自动检测网络配置"""
        # 检查环境变量
        proxy_url = os.getenv('HTTP_PROXY') or os.getenv('HTTPS_PROXY')
        
        config = NetworkConfig()
        
        if proxy_url:
            try:
                parsed = urlparse(proxy_url)
                config.use_proxy = True
                config.proxy_host = parsed.hostname
                config.proxy_port = parsed.port
                config.proxy_username = parsed.username
                config.proxy_password = parsed.password
            except Exception:
                print(f"[network] Invalid proxy URL: {proxy_url}")
        
        # 检查其他环境变量
        config.timeout = int(os.getenv('NETWORK_TIMEOUT', str(config.timeout)))
        config.verify_ssl = os.getenv('VERIFY_SSL', 'true').lower() != 'false'
        
        return config
    
    def _create_session(self) -> requests.Session:
        """创建配置好的会话"""
        session = requests.Session()
        
        # 设置超时
        session.timeout = self.config.timeout
        
        # 设置用户代理
        session.headers.update({
            'User-Agent': self.config.user_agent,
            'Accept': 'application/json, text/plain, */*',
            'Accept-Encoding': 'gzip, deflate',
        })
        
        # 设置代理
        if self.config.use_proxy and self.config.proxy_host:
            proxy_url = f"http://"
            if self.config.proxy_username and self.config.proxy_password:
                proxy_url += f"{self.config.proxy_username}:{self.config.proxy_password}@"
            proxy_url += f"{self.config.proxy_host}"
            if self.config.proxy_port:
                proxy_url += f":{self.config.proxy_port}"
            
            session.proxies = {
                'http': proxy_url,
                'https': proxy_url,
            }
        
        # SSL 验证
        session.verify = self.config.verify_ssl
        
        return session
    
    def test_connectivity(self, url: str = "https://www.google.com") -> bool:
        """测试网络连接"""
        try:
            response = self.session.get(url, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"[network] Connectivity test failed: {e}")
            return False
    
    def get_best_mirror(self, mirrors: List[str], test_path: str = "/") -> str:
        """从镜像列表中选择最快的"""
        best_mirror = mirrors[0]
        best_time = float('inf')
        
        for mirror in mirrors:
            try:
                start_time = time.time()
                response = self.session.get(mirror + test_path, timeout=5)
                elapsed = time.time() - start_time
                
                if response.status_code == 200 and elapsed < best_time:
                    best_mirror = mirror
                    best_time = elapsed
                    
            except Exception:
                continue
        
        print(f"[network] Selected best mirror: {best_mirror} ({best_time:.2f}s)")
        return best_mirror
    
    def download_with_progress(self, url: str, dest_path, chunk_size: int = 8192):
        """带进度显示的下载"""
        try:
            response = self.session.get(url, stream=True, timeout=self.config.timeout)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # 显示进度
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r[download] {progress:.1f}% ({downloaded}/{total_size} bytes)", end='', flush=True)
            
            print()  # 换行
            return True
            
        except Exception as e:
            print(f"\n[download] Failed: {e}")
            return False
    
    def get_response_time(self, url: str) -> float:
        """获取响应时间"""
        try:
            start_time = time.time()
            response = self.session.get(url, timeout=10)
            elapsed = time.time() - start_time
            return elapsed if response.status_code == 200 else float('inf')
        except Exception:
            return float('inf')
    
    def optimize_dns(self):
        """优化 DNS 设置"""
        try:
            # 测试常见 DNS 服务器
            dns_servers = [
                '8.8.8.8',      # Google
                '1.1.1.1',      # Cloudflare
                '208.67.222.222', # OpenDNS
                '9.9.9.9',       # Quad9
            ]
            
            best_dns = dns_servers[0]
            best_time = float('inf')
            
            for dns in dns_servers:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(2)
                    start = time.time()
                    result = sock.connect_ex((dns, 53))
                    elapsed = time.time() - start
                    sock.close()
                    
                    if result == 0 and elapsed < best_time:
                        best_dns = dns
                        best_time = elapsed
                        
                except Exception:
                    continue
            
            print(f"[network] Best DNS: {best_dns} ({best_time:.3f}s)")
            
        except Exception as e:
            print(f"[network] DNS optimization failed: {e}")
    
    def get_network_info(self) -> Dict[str, Any]:
        """获取网络信息"""
        info = {
            'timeout': self.config.timeout,
            'use_proxy': self.config.use_proxy,
            'verify_ssl': self.config.verify_ssl,
            'user_agent': self.config.user_agent,
        }
        
        if self.config.use_proxy:
            info['proxy'] = {
                'host': self.config.proxy_host,
                'port': self.config.proxy_port,
                'has_auth': bool(self.config.proxy_username),
            }
        
        # 测试连接
        info['connectivity'] = self.test_connectivity()
        
        return info


# 全局网络优化器实例
network_optimizer = NetworkOptimizer()
