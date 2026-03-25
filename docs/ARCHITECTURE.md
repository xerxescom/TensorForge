# TensorForge 架构文档

面向开发者的深度架构设计文档。

> 👤 **用户快速上手** 请查看 [`README.md`](../README.md)

## 📁 目录结构

```
TensorForge/
├── src/                         # 源代码
│   ├── benchmarks/              # 测试实现
│   │   ├── llm.py              # LLMBenchmark, LLMContextScaleBenchmark
│   │   ├── multimodal.py       # MultimodalBenchmark, CrossModalBenchmark  
│   │   └── suite.py            # ConcurrentStressTest
│   ├── core/                    # 基础设施
│   │   ├── collector.py        # GPUSampler, BenchmarkRunner, 数据结构
│   │   ├── config_manager.py   # BenchmarkConfig, ConfigManager
│   │   ├── error_handler.py    # ErrorType, ErrorHandler
│   │   ├── model_manager.py    # 模型下载缓存
│   │   └── network_optimizer.py # 镜像源切换
│   ├── reporting/               # HTML报告生成
│   ├── tools/                   # 安装/检查脚本
│   └── cli.py                   # CLI入口
├── examples/                    # 使用示例
├── tests/                       # 测试
├── docs/                        # 本文档
└── config.yaml                  # 配置文件
```

## 🔧 核心组件详解

### 1. GPUSampler - GPU 采样器

**设计目标**: 后台持续采集 GPU 状态，与测试任务解耦

**关键特性**:
- **自适应采样**: 高负载时(>80%)加速采样，低负载时(<20%)降速节省资源
- **智能降采样**: `smart_sampling()` 保留峰值/谷值，控制内存使用
- **跨平台支持**: Windows 自动搜索 DriverStore 中的 nvidia-smi
- **线程安全**: 采样线程与主线程通过锁隔离

**数据流**:
```
nvidia-smi → GPUSample → 内存队列 → smart_sampling() → JSON/CSV
```

### 2. BenchmarkRunner - 测试基类

**设计模式**: Template Method 模式

**生命周期**:
```python
run() -> 预热 -> run_task() [子类实现] -> 停止采样 -> 计算统计 -> 保存结果
```

**输出格式**:
- JSON: 完整 `BenchmarkResult` (含原始 samples)
- CSV: `summary.csv` 追加模式，便于批量分析

### 3. 配置系统

**降级策略**: PyYAML 不存在时自动使用 JSON

**配置层级**:
1. 代码默认值 (BenchmarkConfig 字段)
2. 配置文件 (config.yaml)
3. 运行时参数 (构造函数传入)

### 4. 错误处理

**错误分类**:
- NETWORK: 连接/下载失败 → 重试
- TIMEOUT: 超时 → 重试
- SYSTEM: GPU/CUDA 错误 → 不重试
- CONFIG/MODEL: 配置/模型错误 → 不重试

**重试策略**: 指数退避 + 抖动 (jitter)
```
delay = base_delay * (2 ^ attempt) * random(0.5, 1.0)
```

## 🎯 基准测试设计

### LLM 测试

**LLMBenchmark**:
- Ollama CLI 调用，解析 JSON 输出
- TTFT 估算: 基于总时间 × 0.1 (简化模型)
- Token 估算: `len(response.split())` (简化 tokenizer)

**LLMContextScaleBenchmark**:
- 构造不同长度提示词测试性能衰减
- 基准: 512 tokens，对比 1024/2048/4096

### 多模态测试

当前为**模拟实现** (sleep 模拟推理时间):
- `MultimodalBenchmark`: 并行运行 text/vision/audio
- `CrossModalBenchmark`: 序列化模态切换，测量切换开销

### 并发压测

**ConcurrentStressTest**:
```
1. 基线测试: 单任务顺序执行获取基准
2. 并发测试: 多线程 + Barrier 同步启动
3. 对比分析: 计算吞吐量衰减率和效率比
```

**关键指标**:
- `throughput_degradation`: (baseline - concurrent) / baseline
- `efficiency_ratio`: concurrent / baseline
- `time_variance`: 任务时间方差（调度公平性）

## 📊 数据模型

### GPUSample

```python
@dataclass
class GPUSample:
    timestamp: float
    gpu_util: float          # GPU 利用率 (%)
    memory_used_mb: float    # 显存使用 (MB)
    memory_total_mb: float   # 显存总量 (MB)
    power_w: float          # 功耗 (W)
    temp_c: float           # 温度 (°C)
    clock_mhz: float        # 核心频率 (MHz)
    memory_clock_mhz: float  # 显存频率 (MHz)
```

### BenchmarkResult

```python
@dataclass
class BenchmarkResult:
    task_name: str
    model_name: str
    precision: str
    status: str              # "ok", "error", "timeout"
    duration_s: float
    metrics: Dict[str, Any]   # 业务指标 (子类定义)
    gpu_stats: Dict[str, Any] # GPU 统计 (基类计算)
    raw_samples: List[GPUSample]
    error: Optional[str]
    timestamp: str
```

## 🔌 扩展指南

### 添加新基准测试

```python
# src/benchmarks/my_benchmark.py
from ..core.collector import BenchmarkRunner

class MyBenchmark(BenchmarkRunner):
    def __init__(self, my_param: str, **kwargs):
        super().__init__(task_name="my_task", **kwargs)
        self.my_param = my_param
    
    def run_task(self) -> dict:
        # 实现测试逻辑
        # 返回值会存入 result.metrics
        return {"score": 100.0}
```

注册到包:
```python
# src/__init__.py
from .benchmarks.my_benchmark import MyBenchmark
__all__ = [..., "MyBenchmark"]
```

添加 CLI:
```python
# src/cli.py
from .benchmarks.my_benchmark import MyBenchmark

def run_my(args):
    bench = MyBenchmark(my_param=args.param, **vars(args))
    result = bench.run()
    print(f"Score: {result.metrics['score']}")

parser = subparsers.add_parser("my", help="我的测试")
parser.add_argument("--param", required=True)
parser.set_defaults(func=run_my)
```

## 🧪 测试架构 (计划中)

```
tests/
├── unit/              # 单元测试 (mock GPU)
│   ├── test_collector.py
│   └── test_config.py
├── integration/       # 集成测试 (需真实 GPU)
│   └── test_end_to_end.py
└── performance/       # 性能回归测试
    └── test_scalability.py
```

标记:
- `@pytest.mark.gpu`: 需要 GPU
- `@pytest.mark.slow`: 耗时 >30s

## � 部署

### 开发环境

```bash
pip install -e .       # 可编辑安装
python -m pytest       # 运行测试
black src/ isort src/  # 代码格式化
```

### Python API 使用

```python
from src import LLMBenchmark, GPUSampler

# 完整测试
bench = LLMBenchmark(model_name="llama3.1:8b", n_runs=5)
result = bench.run()

# 仅监控 GPU
sampler = GPUSampler(interval_s=1.0)
sampler.start()
# ... 自定义代码 ...
samples = sampler.stop()
stats = sampler.get_stats()
```

## 🔮 演进路线

### 短期
1. **真实模型推理**: 替换 sleep 模拟
   - Diffusers (Stable Diffusion)
   - Ultralytics (YOLO)
   - Faster-Whisper (ASR)

### 中期
2. **异步架构**: asyncio 替代 threading
3. **多 GPU 支持**: 并行测试多个设备

### 长期
4. **分布式测试**: 多节点集群支持
5. **数据库后端**: 历史数据持久化
6. **Web UI**: 可视化测试管理
