# TensorForge - GPU AI Benchmark Suite

🚀 **全自动 GPU AI 能力测试框架**，覆盖 LLM、多模态任务，多阶段采集推理速度、功耗、温度、显存，并输出 JSON、CSV、HTML 报告。

> 📖 **架构详情** 参见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 🐛 **故障排除** 见文末

## 📁 项目结构

```text
TensorForge/
├── src/tensorforge/              # 主包
│   ├── benchmarks/               # LLM/多模态/并发压测
│   ├── core/                     # GPU采样、配置、错误处理
│   ├── reporting/                # HTML报告生成
│   └── tools/                    # 安装/检查脚本
├── examples/                     # 使用示例
├── docs/                         # 架构文档
├── results/                      # 测试结果 (自动生成)
├── config.yaml                   # 配置文件
└── requirements*.txt               # 依赖文件
```

## 🚀 快速开始

### 安装

```bash
python install.py                    # 智能安装
# 或 pip install -r requirements.txt   # 手动安装
```

### 运行测试

```bash
# CLI 方式
tensorforge llm --model llama3.1:8b --runs 5
tensorforge monitor --duration 60

# Python API
python -c "
from tensorforge import LLMBenchmark
bench = LLMBenchmark(model_name='llama3.1:8b')
result = bench.run()
print(f\"Tokens/s: {result.metrics['tokens_per_s_mean']:.1f}\")
"

# 完整测试套件
python run_suite.py --mode full
```

### 生成报告

```bash
python generate_report.py results/final_report.json
```

## ⚙️ 配置

编辑 `config.yaml`：

```yaml
default_settings:
  sample_interval_s: 0.5      # 采样间隔
  warmup_s: 5.0               # 预热时间
  timeout_s: 120              # 超时时间

gpu_thresholds:
  high_utilization: 80.0
  max_temperature: 85.0

models:
  llm:
    default: "llama3.1:8b"
    prompts: [...]
```

## 📊 测试指标

| 类别 | 指标 |
|------|------|
| **LLM** | tokens/s, TTFT, 成功率, tokens/Joule |
| **多模态** | FPS (视觉), RTF (音频), 切换开销 |
| **并发压测** | 吞吐量衰减, 效率比, 任务时间方差 |
| **系统** | GPU利用率, 显存, 功耗, 温度 |

## �️ 扩展

```python
from tensorforge import BenchmarkRunner

class MyBenchmark(BenchmarkRunner):
    def run_task(self) -> dict:
        return {"metric": value}
```

完整扩展指南见 [ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 🐛 故障排除

| 问题 | 解决 |
|------|------|
| nvidia-smi 找不到 | Windows: `where nvidia-smi` / Linux: `which nvidia-smi` |
| Ollama 连接失败 | `ollama list` 检查服务；重启 Ollama 服务 |
| 模型下载失败 | `python install.py --mirror hf-mirror.com` |

调试模式：`python run_suite.py --verbose`

## 📄 许可证

MIT License
