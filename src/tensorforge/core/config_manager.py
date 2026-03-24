"""
配置管理器 - 支持从 YAML 文件加载配置
"""
import json
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass

# 尝试导入 yaml，如果失败则使用 json
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("[warn] PyYAML not found, using JSON config only")


@dataclass
class BenchmarkConfig:
    """基准测试配置类"""
    sample_interval_s: float = 0.5
    warmup_s: float = 5.0
    timeout_s: int = 120
    max_raw_samples: int = 1000
    adaptive_sampling: bool = True
    
    # GPU 阈值
    high_utilization: float = 80.0
    low_utilization: float = 20.0
    max_temperature: float = 85.0
    max_power_draw: float = 450.0
    
    # 模型配置
    llm_prompts: List[str] = None
    diffusion_n_images: int = 10
    diffusion_n_steps: int = 20
    cv_n_frames: int = 200
    
    def __post_init__(self):
        if self.llm_prompts is None:
            self.llm_prompts = [
                "Explain the difference between a transformer and an RNN in detail.",
                "Write a Python function to compute Fibonacci numbers recursively with memoization.",
                "Describe the water cycle in 300 words.",
                "What are the main causes and consequences of the French Revolution?",
                "Explain how gradient descent works in machine learning.",
            ]


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config = None
    
    def load_config(self) -> BenchmarkConfig:
        """从配置文件加载配置"""
        if self._config is None:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    if HAS_YAML and self.config_path.suffix.lower() in ['.yaml', '.yml']:
                        data = yaml.safe_load(f)
                    else:
                        # 尝试 JSON 格式
                        try:
                            data = json.load(f)
                        except json.JSONDecodeError:
                            print(f"[warn] Config file {self.config_path} not valid JSON/YAML, using defaults")
                            data = {}
                    
                    self._config = self._dict_to_config(data)
            else:
                print(f"[warn] Config file {self.config_path} not found, using defaults")
                self._config = BenchmarkConfig()
        return self._config
    
    def _dict_to_config(self, data: Dict[str, Any]) -> BenchmarkConfig:
        """将字典转换为配置对象"""
        default_settings = data.get('default_settings', {})
        gpu_thresholds = data.get('gpu_thresholds', {})
        models = data.get('models', {})
        
        # LLM prompts
        llm_prompts = []
        if models and 'llm' in models and 'prompts' in models['llm']:
            llm_prompts = models['llm']['prompts']
        
        return BenchmarkConfig(
            sample_interval_s=default_settings.get('sample_interval_s', 0.5),
            warmup_s=default_settings.get('warmup_s', 5.0),
            timeout_s=default_settings.get('timeout_s', 120),
            max_raw_samples=default_settings.get('max_raw_samples', 1000),
            adaptive_sampling=default_settings.get('adaptive_sampling', True),
            
            high_utilization=gpu_thresholds.get('high_utilization', 80.0),
            low_utilization=gpu_thresholds.get('low_utilization', 20.0),
            max_temperature=gpu_thresholds.get('max_temperature', 85.0),
            max_power_draw=gpu_thresholds.get('max_power_draw', 450.0),
            
            llm_prompts=llm_prompts,
            diffusion_n_images=models.get('diffusion', {}).get('n_images', 10),
            diffusion_n_steps=models.get('diffusion', {}).get('n_steps', 20),
            cv_n_frames=models.get('cv', {}).get('n_frames', 200),
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
            }
        }
        
        # 根据文件扩展名选择格式
        if HAS_YAML and self.config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
        else:
            # 保存为 JSON
            json_path = self.config_path.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            print(f"Default config saved to {json_path}")
            return
        
        print(f"Default config saved to {self.config_path}")


# 全局配置实例
config_manager = ConfigManager()
