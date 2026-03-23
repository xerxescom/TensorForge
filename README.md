# TensorForge - GPU AI Benchmark Suite

🚀 **全自动 GPU AI 能力测试框架**，覆盖 LLM、图像生成、CV、ASR，多阶段采集推理速度、功耗、温度、显存，并输出 JSON、CSV、HTML 报告。

## 📁 项目结构

```text
TensorForge/
├── src/                              # 源代码包
│   └── tensorforge/                  # 主包
│       ├── __init__.py
│       ├── benchmarks/               # 基准测试模块
│       │   ├── __init__.py
│       │   ├── llm.py               # LLM 基准测试 (Ollama)
│       │   ├── multimodal.py        # 多模态基准测试 (Diffusion, CV, ASR)
│       │   └── suite.py             # 测试套件控制器
│       ├── core/                     # 核心基础功能
│       │   ├── __init__.py
│       │   ├── collector.py         # GPU 数据采集器
│       │   ├── config_manager.py    # 配置管理器
│       │   ├── error_handler.py     # 错误处理和重试机制
│       │   ├── logging_utils.py     # 日志工具
│       │   ├── model_manager.py     # 模型下载和缓存管理
│       │   ├── network_optimizer.py # 网络优化和镜像切换
│       │   └── tf_logger.py         # 日志记录器
│       ├── reporting/                # 报告生成模块
│       │   ├── __init__.py
│       │   └── generate_report.py   # HTML 报告生成器
│       └── tools/                    # 工具脚本
│           ├── __init__.py
│           ├── check_deps.py        # 依赖检查工具
│           ├── install.py           # 安装脚本
│           └── test_models.py       # 模型测试工具
├── check_deps.py                     # 根目录依赖检查入口
├── generate_report.py                # 根目录报告生成入口
├── install.py                        # 根目录安装入口
├── run_suite.py                      # 根目录测试套件入口
├── test_models.py                    # 根目录模型测试入口
├── config.yaml                       # 配置文件
├── requirements.txt                  # 完整依赖列表
├── requirements-core.txt             # 核心依赖
├── requirements-dev.txt              # 开发依赖
├── requirements-ml.txt               # 机器学习依赖
├── models_cache/                     # 模型缓存目录
├── results/                          # 测试结果目录
│   ├── rtx4070/                     # RTX 4070 测试结果
│   └── rtx5060Ti/                   # RTX 5060 Ti 测试结果
├── .git/                            # Git 版本控制
├── .gitignore                       # Git 忽略文件
├── .idea/                           # IDE 配置
├── __pycache__/                     # Python 缓存
├── yolov8n.pt                       # YOLOv8 模型文件
└── README.md                        # 项目说明文档
```

## 🧭 模块职责说明

### 📊 `src/tensorforge/benchmarks/` - 基准测试模块

负责各种 AI 工作负载的性能测试：

- **`llm.py`** - LLM 基准测试
  - 支持 Ollama 本地模型
  - 测试 tokens/s、TTFT、上下文扩展
  - 能效计算 (tokens/Joule)

- **`multimodal.py`** - 多模态基准测试
  - Diffusion 图像生成 (Stable Diffusion)
  - 计算机视觉 (YOLO 目标检测)
  - 语音识别 (Whisper ASR)

- **`suite.py`** - 测试套件控制器
  - 多阶段测试编排
  - 并发压测
  - CLI 接口

### 🔧 `src/tensorforge/core/` - 核心基础功能

提供底层支持和通用功能：

- **`collector.py`** - GPU 数据采集器
  - 实时 GPU 指标采集 (利用率、功耗、温度、显存)
  - 自适应采样频率
  - 跨平台 nvidia-smi 支持

- **`config_manager.py`** - 配置管理器
  - YAML 配置文件解析
  - 默认配置生成
  - 配置验证

- **`model_manager.py`** - 模型管理器
  - 智能模型下载和缓存
  - 多镜像源支持
  - 断点续传

- **`network_optimizer.py`** - 网络优化器
  - 自动镜像切换
  - 连接测试
  - 代理支持

- **`error_handler.py`** - 错误处理器
  - 错误分类和重试
  - 统计报告
  - 指数退避策略

- **`logging_utils.py`** & **`tf_logger.py`** - 日志系统
  - 统一日志配置
  - 多级别日志输出
  - 文件和控制台双输出

### 📈 `src/tensorforge/reporting/` - 报告生成模块

- **`generate_report.py`** - HTML 报告生成器
  - 现代化 HTML 报告
  - 交互式图表 (Chart.js)
  - 响应式设计
  - 性能指标可视化

### 🛠️ `src/tensorforge/tools/` - 工具脚本

- **`check_deps.py`** - 依赖检查工具
- **`install.py`** - 智能安装脚本
- **`test_models.py`** - 模型测试和验证

## 🚀 快速开始

### 1. 环境检查

```bash
# 检查系统依赖和 GPU 环境
python check_deps.py
```

### 2. 安装依赖

#### 智能安装 (推荐)
```bash
python install.py
```

#### 手动安装
```bash
# 安装完整依赖
pip install -r requirements.txt

# 或分步安装
pip install -r requirements-core.txt
pip install -r requirements-ml.txt
pip install -r requirements-dev.txt
```

### 3. 准备模型

```bash
# 下载 LLM 模型 (可选，支持在线下载)
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
```

### 4. 运行基准测试

```bash
# 运行完整测试套件
python run_suite.py --gpu-name "RTX 4070" --output-dir results/rtx4070

# 只运行特定测试阶段
python run_suite.py --only llm llm_fp16 llm_context_scale

# 跳过耗时测试
python run_suite.py --skip diffusion concurrent

# 自定义配置
python run_suite.py --config custom_config.yaml --verbose
```

### 5. 生成报告

```bash
# 生成 HTML 报告
python generate_report.py results/rtx4070/final_report.html

# 指定输出路径
python generate_report.py results/rtx4070/final_report.json reports/custom_report.html
```

## ⚙️ 配置说明

### `config.yaml` 主要配置项

```yaml
# GPU 配置
gpu:
  index: 0                    # GPU 设备索引
  sampling_interval: 0.5      # 采样间隔 (秒)

# 网络配置
network:
  timeout: 60                 # 请求超时 (秒)
  max_retries: 3              # 最大重试次数
  preferred_endpoints:        # 首选端点
    - "https://huggingface.co"
    - "https://hf-mirror.com"

# 错误处理
error:
  enable_retries: true        # 启用重试
  max_retries: 3              # 最大重试次数
  report_errors: true         # 错误报告

# 缓存配置
cache:
  max_size_gb: 50            # 最大缓存大小 (GB)
  retention_days: 30          # 保留天数
```

## 📊 测试指标

### LLM 测试
- **吞吐量**: tokens/second
- **延迟**: Time To First Token (TTFT)
- **上下文扩展**: 不同上下文长度的性能衰减
- **能效**: tokens/Joule

### 图像生成 (Diffusion)
- **迭代速度**: iterations/second
- **图像生成速度**: images/second
- **功耗效率**: steps/Joule

### 计算机视觉 (CV)
- **推理速度**: FPS (Frames Per Second)
- **延迟**: 单帧推理时间

### 语音识别 (ASR)
- **实时因子**: Real-Time Factor
- **准确率**: Word Error Rate

### 系统指标
- **GPU 利用率**: %
- **显存使用**: MB
- **功耗**: Watts
- **温度**: °C

## 🎯 报告解读

### HTML 报告包含
- **关键性能指标** - 顶部汇总卡片
- **吞吐量对比** - 各任务性能柱状图
- **功耗分析** - 平均和峰值功耗
- **温度监控** - 峰值温度统计
- **能效分析** - 性能/功耗比
- **上下文扩展** - LLM 上下文长度衰减曲线
- **能力雷达图** - 综合性能评分
- **并发性能** - 多任务并发衰减分析

### 性能等级
- 🟢 **优秀** (≥90%) - 性能损失很小
- 🟡 **良好** (70-90%) - 轻微性能损失
- 🔴 **需关注** (<70%) - 明显性能损失

## 🌍 平台兼容性

| 功能 | Windows | Linux | 备注 |
|------|---------|-------|------|
| GPU 采样 | ✅ | ✅ | 自动检测 nvidia-smi |
| Ollama | ✅ | ✅ | 支持本地 LLM |
| Diffusers | ✅ | ✅ | 图像生成 |
| Ultralytics | ✅ | ✅ | YOLO 目标检测 |
| Whisper | ✅ | ✅ | 语音识别 |
| HTML 报告 | ✅ | ✅ | 现代浏览器 |

## � 开发指南

### 添加新的基准测试

1. 在 `src/tensorforge/benchmarks/` 创建新文件
2. 继承 `BenchmarkRunner` 基类
3. 实现 `run_task()` 方法
4. 在 `suite.py` 中注册新测试

```python
from tensorforge.core.collector import BenchmarkRunner

class CustomBenchmark(BenchmarkRunner):
    def run_task(self) -> dict:
        # 实现测试逻辑
        return {"custom_metric": value}
```

### 扩展报告功能

1. 修改 `src/tensorforge/reporting/generate_report.py`
2. 添加新的图表类型
3. 更新 HTML 模板和 JavaScript

### 配置新参数

1. 编辑 `config.yaml` 添加新配置项
2. 在 `config_manager.py` 中添加解析逻辑
3. 在相应模块中使用配置

## 🐛 故障排除

### 常见问题

**Q: nvidia-smi 找不到**
```bash
# Windows 检查安装路径
where nvidia-smi

# Linux 检查 PATH
which nvidia-smi
```

**Q: 模型下载失败**
```bash
# 检查网络连接
python -c "import urllib.request; urllib.request.urlopen('https://huggingface.co')"

# 使用镜像
python install.py --mirror hf-mirror.com
```

**Q: Ollama 连接失败**
```bash
# 检查 Ollama 服务
ollama list

# 重启 Ollama
# Windows: 服务管理器
# Linux: sudo systemctl restart ollama
```

### 调试模式

```bash
# 启用详细日志
python run_suite.py --verbose --debug

# 查看日志文件
tail -f results/*/tensorforge.log
```

## 📝 更新日志

### v2.0.0 - 结构重组
- ✨ 重构为标准 Python 包结构
- 🎨 全新 HTML 报告设计
- 📊 增强日志系统
- 🔧 改进错误处理机制
- 🌐 优化网络下载和镜像支持

### v1.x.x - 初始版本
- 🚀 基础基准测试功能
- 📈 GPU 数据采集
- 📋 报告生成

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [Ollama](https://ollama.com/) - 本地 LLM 运行
- [Hugging Face](https://huggingface.co/) - 模型和数据集
- [Chart.js](https://www.chartjs.org/) - 图表可视化
- [Ultralytics](https://ultralytics.com/) - YOLO 目标检测
- [OpenAI Whisper](https://github.com/openai/whisper) - 语音识别

---

🚀 **TensorForge** - 让 GPU 性能测试变得简单而专业
