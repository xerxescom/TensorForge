"""
配置管理器 - 支持从 YAML 文件加载配置
"""
import yaml
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, field

from tf_logger import logger


@dataclass
class BenchmarkConfig:
    """基准测试配置类"""
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
    network_preferred_endpoints: List[str] = field(
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
    
    # GPU 阈值
    high_utilization: float = 80.0
    low_utilization: float = 20.0
    max_temperature: float = 85.0
    max_power_draw: float = 450.0
    
    # 模型配置
    llm_prompts: List[str] = field(
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
    
    def __post_init__(self):
        # 兼容调用方传入 None 的情况，保持向后兼容
        if self.llm_prompts is None:
            self.llm_prompts = [
                "Explain the difference between a transformer and an RNN in detail.",
                "Write a Python function to compute Fibonacci numbers recursively with memoization.",
                "Describe the water cycle in 300 words.",
                "What are the main causes and consequences of the French Revolution?",
                "Explain how gradient descent works in machine learning.",
            ]
        if self.network_preferred_endpoints is None:
            self.network_preferred_endpoints = [
                "https://huggingface.co",
                "https://hf-mirror.com",
            ]


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config = None
    
    def load_config(self) -> BenchmarkConfig:
        """从 YAML 文件加载配置"""
        if self._config is None:
            if self.config_path.exists():
                try:
                    with open(self.config_path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                except yaml.YAMLError as e:
                    logger.warning(f"Config file {self.config_path} YAML parse failed: {e}; using defaults")
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
    
    def _dict_to_config(self, data: Dict[str, Any]) -> BenchmarkConfig:
        """将字典转换为配置对象"""
        default_settings = data.get('default_settings', {})
        gpu_thresholds = data.get('gpu_thresholds', {})
        models = data.get('models', {})
        network = data.get('network', {})
        cache = data.get('cache', {})
        error_handling = data.get('error_handling', {})
        llm_model_cfg = models.get('llm', {})
        diffusion_model_cfg = models.get('diffusion', {})
        cv_model_cfg = models.get('cv', {})
        asr_model_cfg = models.get('asr', {})
        concurrent_test = data.get('concurrent_test', {})
        
        # LLM prompts
        llm_prompts = None
        if models and 'llm' in models and 'prompts' in models['llm']:
            llm_prompts = models['llm']['prompts']
        
        return BenchmarkConfig(
            sample_interval_s=default_settings.get('sample_interval_s', 0.5),
            warmup_s=default_settings.get('warmup_s', 5.0),
            timeout_s=default_settings.get('timeout_s', 120),
            max_raw_samples=default_settings.get('max_raw_samples', 1000),
            adaptive_sampling=default_settings.get('adaptive_sampling', True),
            cache_base_dir=cache.get('base_dir', 'models_cache'),
            cache_max_size_gb=cache.get('max_size_gb', 50.0),
            cache_cleanup_old_models=cache.get('cleanup_old_models', True),
            cache_model_retention_days=cache.get('model_retention_days', 30),
            network_timeout=network.get('timeout', 300),
            network_max_retries=network.get('max_retries', 5),
            network_retry_delay=network.get('retry_delay', 2.0),
            network_use_proxy=network.get('use_proxy', False),
            network_proxy_host=network.get('proxy_host'),
            network_proxy_port=network.get('proxy_port'),
            network_verify_ssl=network.get('verify_ssl', True),
            network_enable_hf_transfer=network.get('enable_hf_transfer', True),
            network_preferred_endpoints=network.get(
                'preferred_endpoints',
                ["https://huggingface.co", "https://hf-mirror.com"],
            ),
            error_enable_retries=error_handling.get('enable_retries', True),
            error_max_retries=error_handling.get('max_retries', 3),
            error_retry_backoff_factor=error_handling.get('retry_backoff_factor', 2.0),
            error_classify_errors=error_handling.get('classify_errors', True),
            error_report_errors=error_handling.get('report_errors', True),
            
            high_utilization=gpu_thresholds.get('high_utilization', 80.0),
            low_utilization=gpu_thresholds.get('low_utilization', 20.0),
            max_temperature=gpu_thresholds.get('max_temperature', 85.0),
            max_power_draw=gpu_thresholds.get('max_power_draw', 450.0),
            
            llm_prompts=llm_prompts,
            llm_model=llm_model_cfg.get('default', 'llama3.1:8b'),
            llm_precision=llm_model_cfg.get('precision', 'q4_k_m'),
            llm_backend=llm_model_cfg.get('backend', 'ollama'),
            llm_fp16_precision=llm_model_cfg.get('fp16_precision', 'fp16'),
            diffusion_n_images=diffusion_model_cfg.get('n_images', 10),
            diffusion_n_steps=diffusion_model_cfg.get('n_steps', 20),
            diffusion_model=diffusion_model_cfg.get('default', 'sdxl-turbo'),
            diffusion_precision=diffusion_model_cfg.get('precision', 'fp16'),
            diffusion_local_files_only=diffusion_model_cfg.get('local_files_only', False),
            cv_n_frames=cv_model_cfg.get('n_frames', 200),
            cv_model=cv_model_cfg.get('default', 'yolov8n'),
            cv_precision=cv_model_cfg.get('precision', 'fp16'),
            cv_image_size=cv_model_cfg.get('image_size', 640),
            asr_model=asr_model_cfg.get('default', 'base'),
            asr_precision=asr_model_cfg.get('precision', 'float16'),
            concurrent_duration_s=concurrent_test.get('duration_s', 60),
        )
    
    def save_default_config(self):
        """保存默认配置到文件"""
        default_config = {
            'default_settings': {
                'sample_interval_s': 0.5,
                'warmup_s': 5.0,
                'timeout_s': 120,
                'max_raw_samples': 1000,
                'adaptive_sampling': True
            },
            'network': {
                'timeout': 300,
                'max_retries': 5,
                'retry_delay': 2.0,
                'use_proxy': False,
                'proxy_host': None,
                'proxy_port': None,
                'verify_ssl': True,
                'enable_hf_transfer': True,
                'preferred_endpoints': [
                    'https://huggingface.co',
                    'https://hf-mirror.com'
                ]
            },
            'cache': {
                'base_dir': 'models_cache',
                'max_size_gb': 50,
                'cleanup_old_models': True,
                'model_retention_days': 30
            },
            'gpu_thresholds': {
                'high_utilization': 80.0,
                'low_utilization': 20.0,
                'max_temperature': 85.0,
                'max_power_draw': 450.0
            },
            'models': {
                'llm': {
                    'default': 'llama3.1:8b',
                    'precision': 'q4_k_m',
                    'prompts': [
                        "Explain the difference between a transformer and an RNN in detail.",
                        "Write a Python function to compute Fibonacci numbers recursively with memoization.",
                        "Describe the water cycle in 300 words."
                    ]
                },
                'diffusion': {
                    'default': 'stabilityai/sdxl-turbo',
                    'precision': 'fp16',
                    'n_images': 10,
                    'n_steps': 20
                },
                'cv': {
                    'default': 'yolov8n',
                    'precision': 'fp16',
                    'n_frames': 200,
                    'image_size': 640
                }
            },
            'error_handling': {
                'enable_retries': True,
                'max_retries': 3,
                'retry_backoff_factor': 2.0,
                'classify_errors': True,
                'report_errors': True
            }
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Default config saved to {self.config_path}")


# 全局配置实例
config_manager = ConfigManager()
