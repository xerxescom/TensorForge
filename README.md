# TensorForge - GPU AI Benchmark Suite

🚀 **全自动 GPU AI 能力测试框架**，涵盖 LLM / 图像生成 / CV / 语音，
自动采集推理速度、功耗、温度、显存，输出 JSON + CSV + HTML 报告。

✨ **全新特性**：智能模型缓存、网络优化、错误重试、配置文件支持

**支持系统：Windows 10/11 · Linux（Ubuntu 20.04+）**


## 📁 目录结构

```
TensorForge/
├── 📊 collector.py           # GPU 采样器 + 基准测试基类（跨平台核心）
├── 🤖 llm_bench.py          # LLM 推理 + 上下文长度衰减测试
├── 🎨 other_bench.py        # Diffusion / CV / ASR 测试
├── 📈 generate_report.py    # 生成 HTML 可视化报告
├── 🚀 run_suite.py          # 完整套件入口 + 并发压测
├── ⚙️ config.yaml           # 统一配置文件
├── 📦 model_manager.py      # 智能模型下载和缓存管理
├── 🌐 network_optimizer.py   # 网络连接优化器
├── 🛡️ error_handler.py      # 错误处理和重试机制
├── 🔧 check_deps.py          # 依赖检查脚本
├── 🧪 test_models.py        # 模型测试脚本
├── 💻 install.py             # 智能安装脚本
├── 📄 requirements.txt       # 完整依赖列表
├── 📄 requirements-core.txt  # 核心依赖
├── 📄 requirements-ml.txt     # ML 框架依赖
├── 📄 requirements-dev.txt   # 开发工具依赖
└── 📋 README.md             # 项目说明文档
```

## 🚀 快速开始

### 1. 环境检查

```bash
# 检查所有依赖和系统环境
python check_deps.py
```

### 2. 安装依赖

#### 方法一：智能安装（推荐）
```bash
# 自动检测系统环境并安装合适版本
python install.py
```

#### 方法二：手动安装
```bash
# 安装完整依赖
pip install -r requirements.txt

# 或分步安装
pip install -r requirements-core.txt    # 核心依赖
pip install -r requirements-ml.txt      # ML 框架
pip install -r requirements-dev.txt      # 开发工具（可选）
```

#### 方法三：PyTorch 特定安装
```bash
# 根据你的 CUDA 版本选择
# CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 然后安装其他依赖
pip install -r requirements-ml.txt
```

### 3. LLM 支持（可选）

```bash
# 安装 Ollama
# Windows: 下载 https://ollama.com/download/windows 安装包
# Linux: curl -fsSL https://ollama.com/install.sh | sh

# 下载 LLM 模型
ollama pull llama3.1:8b
```

### 4. 测试安装

```bash
# 测试模型下载和管理
python test_models.py

# 验证依赖安装
python check_deps.py
```

### 5. 运行完整测试套件

```bash
# Windows（PowerShell 或 CMD）
python run_suite.py --gpu-name "RTX 4070" --output-dir results\rtx4070

# Linux
python run_suite.py --gpu-name "RTX 4070" --output-dir results/rtx4070

# 只测 LLM
python run_suite.py --only llm llm_fp16 llm_context_scale

# 跳过耗时的 Diffusion
python run_suite.py --skip diffusion concurrent
```

### 6. 生成 HTML 报告

```bash
# Windows
python generate_report.py results\rtx4090\final_report.json

# Linux
python generate_report.py results/rtx4090/final_report.json
```

浏览器直接打开生成的 `report.html`，无需服务器。

---

## 📦 依赖管理

TensorForge 提供了灵活的依赖管理方案：

### Requirements 文件说明

| 文件 | 描述 | 用途 |
|------|------|------|
| `requirements.txt` | 完整依赖 | 包含所有功能所需的包 |
| `requirements-core.txt` | 核心依赖 | 最小化安装，基础功能 |
| `requirements-ml.txt` | ML 框架 | 机器学习和深度学习库 |
| `requirements-dev.txt` | 开发工具 | 测试、代码质量、文档 |

### 安装选项

#### 🚀 智能安装（推荐新手）
```bash
python install.py
```
- 自动检测 CUDA 版本
- 选择合适的 PyTorch 版本
- 检查系统工具
- 提供详细反馈

#### 📦 分步安装（推荐高级用户）
```bash
# 1. 核心依赖（必需）
pip install -r requirements-core.txt

# 2. ML 框架（按需）
pip install -r requirements-ml.txt

# 3. 开发工具（可选）
pip install -r requirements-dev.txt
```

#### 🔧 手动安装（完全控制）
```bash
# 根据你的 CUDA 版本选择 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 然后安装其他依赖
pip install transformers diffusers accelerate huggingface_hub
pip install ultralytics faster-whisper soundfile
pip install requests PyYAML psutil tqdm
```

---

## ✨ 核心特性

### 🚀 **智能模型管理**
- **自动下载**: 支持 Hugging Face 模型自动下载
- **多镜像支持**: 主站 + 中国镜像站，自动切换最佳连接
- **断点续传**: 大文件续传，避免重新下载
- **智能缓存**: 本地缓存模型，避免重复下载
- **并发控制**: 防止重复下载同一模型

### 🌐 **网络优化**
- **自动重试**: 5 次重试机制，指数退避
- **代理支持**: 自动检测和使用系统代理
- **连接测试**: 自动选择最佳下载端点
- **SSL 配置**: 灵活的 SSL 验证设置

### 📊 **动态采样**
- **自适应频率**: 根据 GPU 负载动态调整采样频率
- **智能内存管理**: 保留关键数据点，减少内存占用
- **健康监控**: 实时监控采样器健康状态

### 🛡️ **错误处理**
- **错误分类**: 精确识别错误类型（超时、OOM、依赖缺失等）
- **重试策略**: 不同错误类型的智能重试机制
- **错误报告**: 详细的错误统计和分析

### ⚙️ **配置管理**
- **YAML 配置**: 统一的配置文件管理
- **灵活配置**: 网络超时、重试次数、缓存大小等可配置
- **环境优化**: 自动设置最佳环境变量

## 🌍 平台兼容性说明（Windows vs Linux）

| 功能 | Windows | Linux |
|------|---------|-------|
| GPU 采样 (nvidia-smi) | 自动搜索路径 | PATH 直接调用 |
| 子进程（无黑窗口） | CREATE_NO_WINDOW | 默认 |
| ollama LLM | 支持（需安装包） | 支持（脚本安装） |
| Python 子进程 | sys.executable（路径安全） | sys.executable |
| diffusers / ultralytics | 支持 | 支持 |
| faster-whisper | 支持 | 支持 |
| 模型缓存管理 | 支持 | 支持 |
| 网络优化 | 支持 | 支持 |

### Windows 已知限制

**nvidia-smi 不在 PATH** 时框架会自动搜索以下位置：
```
C:\Windows\System32\nvidia-smi.exe
C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe
C:\Windows\System32\DriverStore\FileRepository\nv_dispi.inf_amd64_*\
```
如果仍找不到，手动将 nvidia-smi 所在目录加入系统 PATH。

**功耗读数**：部分笔记本 GPU（MX 系列、某些移动版）的 `power.draw` 在 Windows 上可能返回 `[N/A]`，框架会自动忽略该字段，其他指标正常采集。

---

## ⚙️ 配置文件

TensorForge 使用 `config.yaml` 进行统一配置：

```yaml
# 网络和下载配置
network:
  timeout: 300              # 下载超时（秒）
  max_retries: 5            # 最大重试次数
  use_proxy: false          # 是否使用代理
  enable_hf_transfer: true  # 启用 Hugging Face 快速传输

# 模型缓存配置
cache:
  base_dir: "models_cache"  # 缓存目录
  max_size_gb: 50          # 最大缓存大小
  cleanup_old_models: true  # 自动清理旧模型

# GPU 阈值
gpu_thresholds:
  high_utilization: 80.0    # 高负载阈值
  max_temperature: 85.0     # 最大温度
  max_power_draw: 450.0    # 最大功耗

# 模型配置
models:
  diffusion:
    default: "sdxl-turbo"
    huggingface_id: "stabilityai/sdxl-turbo"
    download_timeout: 600   # 下载超时（秒）
```

## 🎯 使用示例

### 单独调用某个测试

```python
from llm_bench import LLMBenchmark
from other_bench import DiffusionBenchmark

# LLM 推理测试
llm_bench = LLMBenchmark(
    model_name="llama3.1:8b",
    precision="q4_k_m",
    n_runs=5,
    output_dir="results",
    warmup_s=5,
)
result = llm_bench.run()

# 图像生成测试（自动下载模型）
diffusion_bench = DiffusionBenchmark(
    model_name="sdxl-turbo",  # 使用简化模型名
    n_images=10,
    n_steps=20,
    output_dir="results",
)
result = diffusion_bench.run()
```

### 模型管理

```python
from model_manager import model_manager

# 下载模型（自动缓存）
model_path = model_manager.get_model_path("sdxl-turbo")
print(f"Model cached at: {model_path}")

# 添加自定义模型
model_manager.add_custom_model(
    "my-model", 
    "username/model-name",
    timeout=600
)

# 清理缓存
model_manager.clear_cache("sdxl-turbo")
```

### 网络优化

```python
from network_optimizer import download_optimizer

# 自动设置网络优化
download_optimizer.setup()

# 测试连接
from network_optimizer import NetworkOptimizer
optimizer = NetworkOptimizer()
connectivity = optimizer.test_connectivity()
best_endpoint = optimizer.get_best_endpoint()
```

## 📊 输出说明

| 文件 | 内容 |
|------|------|
| `results/<task>_<model>_<ts>.json` | 完整数据：指标 + GPU 统计 + 原始采样序列 |
| `results/summary.csv` | 每次测试一行，追加写入，方便跨卡横向对比 |
| `results/final_report.json` | 全套汇总 |
| `results/report.html` | 可视化报告，包含 6 个交互图表 |
| `models_cache/` | 智能模型缓存目录 |

---

## ⚡ GPU 性能控制

### Windows（PowerShell，需管理员权限）
```powershell
# 查看当前 GPU 状态
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv

# 固定功耗上限（示例 RTX 4090 默认 450W）
nvidia-smi -pl 450

# 禁用 GPU Boost（可选，保证时钟稳定）
nvidia-smi --lock-gpu-clocks=2100,2100

# 测试完成后解锁
nvidia-smi --reset-gpu-clocks
```

### Linux
```bash
sudo nvidia-smi -pl 450
sudo nvidia-smi --lock-gpu-clocks=2100,2100
# 完成后
sudo nvidia-smi --reset-gpu-clocks
```

---

## 🐛 故障排除

### 常见问题

#### 1. 安装问题
```bash
# 检查 Python 版本（需要 3.12+）
python --version

# 检查 pip 版本
pip --version

# 升级 pip
pip install --upgrade pip

# 清理 pip 缓存
pip cache purge
```

#### 2. 模型下载超时
```bash
# 检查网络连接
python network_optimizer.py

# 使用代理
# 编辑 config.yaml:
# network:
#   use_proxy: true
#   proxy_host: "127.0.0.1"
#   proxy_port: 7890

# 手动清理缓存重试
python -c "from model_manager import model_manager; model_manager.clear_cache()"

# 使用镜像站
export HF_ENDPOINT=https://hf-mirror.com
python test_models.py
```

#### 3. PyTorch 安装失败
```bash
# 检查 CUDA 版本
nvidia-smi

# 手动选择合适的 PyTorch 版本
# CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### 4. 依赖缺失
```bash
# 检查所有依赖
python check_deps.py

# 重新安装依赖
pip install -r requirements.txt --force-reinstall

# 单独安装缺失的包
pip install package_name
```

#### 5. GPU 监控失败
```bash
# 检查 nvidia-smi
nvidia-smi

# Windows 用户：确保 nvidia-smi 在 PATH 中
# 或安装 NVIDIA 驱动到默认位置

# 检查 GPU 是否可用
python -c "import torch; print(torch.cuda.is_available())"
```

#### 6. Ollama 不可用
```bash
# 检查 Ollama
ollama version

# 重新安装
# Windows: https://ollama.com/download/windows
# Linux: curl -fsSL https://ollama.com/install.sh | sh

# 启动 Ollama 服务
ollama serve
```

### 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 测试单个组件
python test_models.py --debug

# 检查配置
python -c "from config_manager import config_manager; print(config_manager.load_config())"
```

### 环境变量设置

```bash
# 设置 Hugging Face 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 设置缓存目录
export HF_HOME=/path/to/cache
export TRANSFORMERS_CACHE=/path/to/cache

# 禁用遥测
export HF_HUB_DISABLE_TELEMETRY=1

# 启用快速传输
export HF_HUB_ENABLE_HF_TRANSFER=1
```

---

## 🤝 贡献指南

### 开发环境设置

```bash
# 克隆项目
git clone <repository-url>
cd TensorForge

# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
python test_models.py
python check_deps.py

# 代码质量检查（lint）
ruff check .

# 代码格式化
ruff format .

# 类型检查
mypy .

# 运行测试套件
pytest
```

### 常用开发命令（推荐）

```bash
# 一次性跑完：lint + format-check + type-check + tests
ruff check . && ruff format --check . && mypy && pytest
```

### 可选：启用 pre-commit

```bash
pip install -r requirements-dev.txt
pre-commit install
```

### 代码结构

- **collector.py**: 核心 GPU 采样和基准测试基类
- **model_manager.py**: 模型下载和缓存管理
- **network_optimizer.py**: 网络连接优化
- **error_handler.py**: 错误处理和重试机制
- **config_manager.py**: 配置文件管理
- **llm_bench.py**: LLM 推理测试
- **other_bench.py**: 图像生成、CV、语音测试
- **install.py**: 智能安装脚本
- **check_deps.py**: 依赖检查工具

### 添加新测试

1. 继承 `BenchmarkRunner` 基类
2. 实现 `run_task()` 方法
3. 返回业务指标字典
4. 在 `run_suite.py` 中注册新测试

### 代码规范

- 使用 Ruff 进行代码检查与格式化
- 使用 mypy 进行类型检查
- 添加适当的文档字符串
- 遵循 PEP 8 编码规范

### CLI 配置覆盖（可复现实验）

`run_suite.py` 支持从指定配置文件加载，并用命令行覆盖部分字段；最终解析结果会写入输出目录的 `config_resolved.json`。

```bash
python run_suite.py --config config.yaml --set sample_interval_s=0.25 --set network_timeout=600
```

---

## 📈 性能基准

### 典型测试结果（RTX 4090）

| 测试类型 | 指标 | 结果 |
|---------|------|------|
| LLM (LLaMA 3.1 8B) | Tokens/s | ~120 |
| Diffusion (SDXL-Turbo) | Iterations/s | ~8 |
| CV (YOLOv8n) | FPS | ~800 |
| ASR (Whisper Base) | RTF | ~0.1 |

*结果因硬件配置和软件版本而异*

---

## 📄 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

---

## 🙏 致谢

- [Hugging Face](https://huggingface.co) - 模型托管和 diffusers 库
- [Ollama](https://ollama.com) - 本地 LLM 推理
- [Ultralytics](https://ultralytics.com) - YOLO 模型
- [PyTorch](https://pytorch.org) - 深度学习框架

---

**🚀 TensorForge - 让 GPU 基准测试变得简单而强大！**

