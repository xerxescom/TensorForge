# TensorForge - GPU AI Benchmark Suite

🚀 **全自动 GPU AI 能力测试框架**，覆盖 LLM、图像生成、CV、ASR，多阶段采集推理速度、功耗、温度、显存，并输出 JSON、CSV、HTML 报告。

这次整理将原来全部堆在仓库根目录的 Python 代码拆分成了更清晰的包结构，保留了根目录脚本入口，因此原有命令基本不用改。

## 📁 当前目录结构

```text
TensorForge/
├── src/
│   └── tensorforge/
│       ├── benchmarks/          # 任务编排与各类 benchmark 实现
│       │   ├── llm.py
│       │   ├── multimodal.py
│       │   └── suite.py
│       ├── core/                # 核心基础能力
│       │   ├── collector.py
│       │   ├── config_manager.py
│       │   ├── error_handler.py
│       │   ├── logging_utils.py
│       │   ├── model_manager.py
│       │   ├── network_optimizer.py
│       │   └── tf_logger.py
│       ├── reporting/           # 报告生成
│       │   └── generate_report.py
│       └── tools/               # 安装、检查、测试工具脚本
│           ├── check_deps.py
│           ├── install.py
│           └── test_models.py
├── check_deps.py                # 根目录 CLI 入口
├── generate_report.py           # 根目录 CLI 入口
├── install.py                   # 根目录 CLI 入口
├── run_suite.py                 # 根目录 CLI 入口
├── test_models.py               # 根目录 CLI 入口
├── config.yaml
├── requirements.txt
├── requirements-core.txt
├── requirements-ml.txt
├── requirements-dev.txt
├── models_cache/
└── results/
```

## 🧭 模块职责说明

### `src/tensorforge/core/`
放通用基础设施，避免 benchmark 逻辑与底层工具互相穿插。

- `collector.py`：GPU 采样、Benchmark 基类、结果对象。
- `config_manager.py`：统一读取 `config.yaml`。
- `model_manager.py`：模型缓存与下载。
- `network_optimizer.py`：网络与镜像优化。
- `error_handler.py`：错误分类、重试、统计。
- `logging_utils.py` / `tf_logger.py`：日志配置与 fallback logger。

### `src/tensorforge/benchmarks/`
放各类测试任务与总控入口。

- `llm.py`：LLM benchmark、上下文长度衰减测试。
- `multimodal.py`：Diffusion、CV、ASR benchmark。
- `suite.py`：完整套件调度、并发压测、CLI 入口。

### `src/tensorforge/reporting/`
- `generate_report.py`：读取 `final_report.json` 生成 HTML 报告。

### `src/tensorforge/tools/`
- `check_deps.py`：依赖检查。
- `install.py`：安装脚本。
- `test_models.py`：模型下载/缓存验证脚本。

## 🚀 快速开始

### 1. 检查依赖

```bash
python check_deps.py
```

### 2. 安装依赖

#### 方法一：智能安装
```bash
python install.py
```

#### 方法二：手动安装
```bash
pip install -r requirements.txt
```

或拆分安装：

```bash
pip install -r requirements-core.txt
pip install -r requirements-ml.txt
pip install -r requirements-dev.txt
```

### 3. 准备 LLM（可选）

```bash
ollama pull llama3.1:8b
```

### 4. 运行完整测试

```bash
python run_suite.py --gpu-name "RTX 4070" --output-dir results/rtx4070
```

只跑指定阶段：

```bash
python run_suite.py --only llm llm_fp16 llm_context_scale
```

跳过耗时任务：

```bash
python run_suite.py --skip diffusion concurrent
```

### 5. 生成 HTML 报告

```bash
python generate_report.py results/rtx4070/final_report.json
```

## 🧪 如果你要开发或扩展

### 推荐的导入方式

优先通过 `tensorforge` 包导入；现在即使直接在仓库根目录开发，不安装包也可以使用 `from tensorforge...` 方式导入。

```python
from tensorforge.core.collector import BenchmarkRunner
from tensorforge.benchmarks.llm import LLMBenchmark
from tensorforge.reporting.generate_report import generate
```

### 新功能应该放哪里

- 新的 GPU / 系统基础能力：放 `core/`
- 新的 benchmark 类型：放 `benchmarks/`
- 新的报表输出：放 `reporting/`
- 一次性安装、诊断、迁移工具：放 `tools/`

### 兼容性说明

仓库根目录现在只保留真正需要的 CLI 入口脚本：

- `python run_suite.py`
- `python check_deps.py`
- `python install.py`
- `python test_models.py`
- `python generate_report.py`

其余历史模块文件已经移除，避免根目录和 `src/` 下出现双份实现。

## 📦 依赖文件说明

| 文件 | 描述 |
|------|------|
| `requirements.txt` | 完整依赖 |
| `requirements-core.txt` | 核心依赖 |
| `requirements-ml.txt` | 机器学习相关依赖 |
| `requirements-dev.txt` | 开发与测试工具 |

## ✨ 核心能力

- 智能模型下载与缓存
- 网络镜像切换与重试
- 自适应 GPU 采样
- 错误分类与重试机制
- JSON / CSV / HTML 报表输出
- Windows / Linux 双平台支持

## 🌍 平台兼容性

| 功能 | Windows | Linux |
|------|---------|-------|
| `nvidia-smi` 采样 | 自动搜索路径 | PATH 调用 |
| Ollama | 支持 | 支持 |
| diffusers / ultralytics / whisper | 支持 | 支持 |
| 模型缓存管理 | 支持 | 支持 |
| HTML 报表输出 | 支持 | 支持 |

## 📌 迁移提示

如果你之前习惯在根目录找实现代码，现在请直接进入 `src/tensorforge/`；同时也可以直接在仓库根目录运行 `from tensorforge...` 导入：

- 采样与基础设施：`src/tensorforge/core/`
- benchmark 实现：`src/tensorforge/benchmarks/`
- 报表：`src/tensorforge/reporting/`
- 工具脚本实现：`src/tensorforge/tools/`

这样后续继续扩展时会更容易维护，也不会再遇到“根目录和包目录各有一份代码”的困惑。
