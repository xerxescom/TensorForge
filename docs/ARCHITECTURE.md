# TensorForge 架构文档

## 🏗️ 项目架构概览

TensorForge 采用模块化架构设计，支持扩展性和可维护性。

## 📁 目录结构

```
TensorForge/
├── src/tensorforge/              # 主包
│   ├── __init__.py              # 包入口
│   ├── cli.py                   # 命令行接口
│   ├── core/                    # 核心模块
│   │   ├── __init__.py
│   │   ├── collector.py         # GPU 采样和基准测试基类
│   │   ├── config_manager.py    # 配置管理
│   │   ├── error_handler.py     # 错误处理
│   │   ├── logging_utils.py     # 日志工具
│   │   ├── model_manager.py     # 模型管理
│   │   ├── network_optimizer.py # 网络优化
│   │   └── tf_logger.py         # 日志记录器
│   ├── benchmarks/              # 基准测试模块
│   │   ├── __init__.py
│   │   ├── llm.py              # LLM 基准测试
│   │   ├── multimodal.py       # 多模态测试
│   │   └── suite.py            # 测试套件
│   ├── tools/                   # 工具模块
│   │   ├── __init__.py
│   │   └── [各种工具]
│   └── reporting/               # 报告生成
│       ├── __init__.py
│       └── [报告工具]
├── examples/                    # 使用示例
│   └── basic_usage.py
├── scripts/                     # 脚本工具
│   └── setup_dev.py
├── docs/                        # 文档
│   ├── ARCHITECTURE.md
│   └── [其他文档]
├── pyproject.toml              # 项目配置
├── requirements*.txt           # 依赖文件
├── config.yaml                 # 配置文件
└── README.md                   # 项目说明
```

## 🔧 核心组件

### 1. GPU 采样器 (GPUSampler)

**职责**: 后台持续采样 GPU 状态

**特性**:
- 智能采样频率调整
- 跨平台 nvidia-smi 支持
- 内存优化的样本存储
- 线程安全设计

**使用流程**:
```python
sampler = GPUSampler(interval_s=0.5, adaptive_sampling=True)
sampler.start()
# ... 运行测试 ...
samples = sampler.stop()
```

### 2. 基准测试基类 (BenchmarkRunner)

**职责**: 统一的基准测试接口

**特性**:
- 自动 GPU 统计集成
- 结果保存和导出
- 预热和超时管理
- 错误处理和重试

**继承关系**:
```
BenchmarkRunner
├── LLMBenchmark
├── DiffusionBenchmark
├── CVBenchmark
└── ASRBenchmark
```

### 3. 配置管理器 (ConfigManager)

**职责**: 统一配置管理

**特性**:
- YAML/JSON 双格式支持
- 优雅的依赖降级
- 默认配置生成
- 配置验证

## 🎯 基准测试架构

### LLM 基准测试流程

```
1. 模型检查
   ├── Ollama 可用性检查
   ├── 模型存在性验证
   └── 自动拉取缺失模型

2. 预热阶段
   ├── GPU 采样器启动
   ├── 模型加载预热
   └── 系统稳定等待

3. 推理测试
   ├── 多轮推理执行
   ├── 性能指标收集
   └── GPU 状态监控

4. 结果分析
   ├── 统计指标计算
   ├── 性能报告生成
   └── 结果文件保存
```

### 性能指标体系

**时间指标**:
- TTFT (Time To First Token)
- Tokens/s (生成速度)
- 响应时间

**资源指标**:
- GPU 利用率
- 显存使用
- 功耗
- 温度

**质量指标**:
- 成功率
- 错误率
- 稳定性

## 🔌 扩展性设计

### 添加新的基准测试

1. **继承 BenchmarkRunner**:
```python
class NewBenchmark(BenchmarkRunner):
    def __init__(self, **kwargs):
        super().__init__(task_name="new_task", **kwargs)
    
    def run_task(self) -> dict:
        # 实现具体的测试逻辑
        return {"metric": value}
```

2. **注册到包**:
```python
# src/tensorforge/__init__.py
from .benchmarks.new_benchmark import NewBenchmark
__all__ = [..., "NewBenchmark"]
```

3. **添加 CLI 支持**:
```python
# src/tensorforge/cli.py
def run_new_benchmark(args):
    bench = NewBenchmark(**vars(args))
    result = bench.run()
```

### 添加新的工具模块

1. **创建模块文件**:
```python
# src/tensorforge/tools/new_tool.py
def new_tool_function():
    pass
```

2. **更新 __init__.py**:
```python
# src/tensorforge/tools/__init__.py
from .new_tool import new_tool_function
```

## 🧪 测试架构

### 测试分层

```
tests/
├── unit/           # 单元测试
│   ├── test_core.py
│   ├── test_config.py
│   └── test_benchmarks.py
├── integration/    # 集成测试
│   ├── test_end_to_end.py
│   └── test_cli.py
└── performance/    # 性能测试
    └── test_scalability.py
```

### 测试标记

- `@pytest.mark.gpu`: 需要 GPU 的测试
- `@pytest.mark.slow`: 耗时较长的测试
- `@pytest.mark.integration`: 集成测试

## 📊 报告系统

### 报告类型

1. **JSON 详细报告**: 完整的原始数据
2. **CSV 汇总报告**: 便于分析的表格数据
3. **HTML 可视化报告**: 图表和可视化
4. **Markdown 总结报告**: 人类友好的总结

### 报告内容

- 测试环境信息
- 性能指标统计
- GPU 资源使用情况
- 错误和异常记录
- 历史对比分析

## 🔒 错误处理策略

### 错误分类

1. **系统错误**: GPU 驱动、内存不足等
2. **网络错误**: 模型下载、API 调用等
3. **模型错误**: 模型加载、推理失败等
4. **配置错误**: 参数错误、文件缺失等

### 处理策略

- **重试机制**: 网络和临时错误
- **降级策略**: 模型不可用时的备选方案
- **用户友好**: 清晰的错误信息和建议
- **日志记录**: 详细的错误上下文

## 🚀 部署架构

### 开发环境

```bash
# 安装开发依赖
pip install -e .[dev]

# 运行测试
pytest

# 代码格式化
black src/ tests/
isort src/ tests/

# 类型检查
mypy src/
```

### 生产环境

```bash
# 安装运行时依赖
pip install tensorforge[ml]

# 运行基准测试
tensorforge llm --model llama3.1:8b

# GPU 监控
tensorforge monitor --duration 300
```

## 🔮 未来扩展

### 计划中的功能

1. **分布式测试**: 多 GPU、多节点支持
2. **云端集成**: AWS、Azure、GCP 支持
3. **自动化 CI/CD**: 持续性能测试
4. **Web 界面**: 图形化测试管理
5. **数据库后端**: 大规模结果存储

### 技术演进

- **异步架构**: 提升并发性能
- **微服务化**: 组件解耦
- **容器化**: Docker/Kubernetes 支持
- **监控集成**: Prometheus/Grafana 集成
