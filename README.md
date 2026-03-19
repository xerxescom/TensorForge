# GPU AI Benchmark Suite

全自动 GPU AI 能力测试框架，涵盖 LLM / 图像生成 / CV / 语音，
自动采集推理速度、功耗、温度、显存，输出 JSON + CSV + HTML 报告。

**支持系统：Windows 10/11 · Linux（Ubuntu 20.04+）**

> macOS 自 2019 年起不再支持 NVIDIA GPU，不在支持范围内。

## 目录结构

```
gpu_benchmark/
├── core/
│   └── collector.py        # GPUSampler + BenchmarkRunner 基类（跨平台核心）
├── tasks/
│   ├── llm_bench.py        # LLM 推理 + 上下文长度衰减
│   └── other_bench.py      # Diffusion / CV / ASR
├── report/
│   └── generate_report.py  # 生成 HTML 可视化报告
└── run_suite.py            # 完整套件入口 + 并发压测
```

## 快速开始

### 1. 安装依赖

```bash
# 核心（必须）——选择适合你 CUDA 版本的命令
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# LLM：ollama（Windows / Linux 均支持）
# Windows: 下载 https://ollama.com/download/windows 安装包
# Linux:   curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b

# 图像生成（可选）
pip install diffusers accelerate transformers

# CV（可选）
pip install ultralytics

# 语音（可选）
pip install faster-whisper soundfile
```

### 2. 运行完整测试套件

```bash
# Windows（PowerShell 或 CMD）
python run_suite.py --gpu-name "RTX 4090" --output-dir results\rtx4090

# Linux
python run_suite.py --gpu-name "RTX 4090" --output-dir results/rtx4090

# 只测 LLM
python run_suite.py --only llm llm_fp16 llm_context_scale

# 跳过耗时的 Diffusion
python run_suite.py --skip diffusion concurrent
```

### 3. 生成 HTML 报告

```bash
# Windows
python report\generate_report.py results\rtx4090\final_report.json

# Linux
python report/generate_report.py results/rtx4090/final_report.json
```

浏览器直接打开生成的 `report.html`，无需服务器。

---

## 平台兼容性说明（Windows vs Linux）

| 功能 | Windows | Linux |
|------|---------|-------|
| GPU 采样 (nvidia-smi) | 自动搜索路径 | PATH 直接调用 |
| 子进程（无黑窗口） | CREATE_NO_WINDOW | 默认 |
| ollama LLM | 支持（需安装包） | 支持（脚本安装） |
| Python 子进程 | sys.executable（路径安全） | sys.executable |
| diffusers / ultralytics | 支持 | 支持 |
| faster-whisper | 支持 | 支持 |

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

## 变量控制清单

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

## 单独调用某个测试

```python
from tasks.llm_bench import LLMBenchmark

bench = LLMBenchmark(
    model_name="llama3.1:8b",
    precision="q4_k_m",
    n_runs=5,
    output_dir="results",
    warmup_s=5,
)
result = bench.run()
# 自动保存到 results/llm_inference_*.json 和 results/summary.csv
```

## 输出说明

| 文件 | 内容 |
|------|------|
| `results/<task>_<model>_<ts>.json` | 完整数据：指标 + GPU 统计 + 原始采样序列 |
| `results/summary.csv` | 每次测试一行，追加写入，方便跨卡横向对比 |
| `results/final_report.json` | 全套汇总 |
| `results/report.html` | 可视化报告，包含 6 个交互图表 |

