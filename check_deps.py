#!/usr/bin/env python3
"""
依赖检查和安装脚本
确保所有必要的依赖都已安装
"""
import importlib
import subprocess
import sys


def check_python_version():
    """检查 Python 版本"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✅ Python version OK")
        return True


def check_package(package_name, import_name=None, min_version=None):
    """检查包是否已安装"""
    import_name = import_name or package_name
    
    try:
        module = importlib.import_module(import_name)
        
        # 检查版本
        if min_version and hasattr(module, '__version__'):
            version = module.__version__
            print(f"✅ {package_name} {version}")
        else:
            print(f"✅ {package_name}")
        
        return True
    
    except ImportError:
        print(f"❌ {package_name} not installed")
        return False


def install_package(package_name, import_name=None):
    """安装包"""
    print(f"Installing {package_name}...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", package_name
        ])
        
        # 验证安装
        import_name = import_name or package_name.split('[')[0].split('==')[0]
        importlib.import_module(import_name)
        print(f"✅ {package_name} installed successfully")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_name}: {e}")
        return False
    except ImportError:
        print(f"❌ {package_name} installation verification failed")
        return False


def check_torch():
    """检查 PyTorch 和 CUDA"""
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            print(f"✅ CUDA {cuda_version} available ({device_count} devices)")
            
            # 显示 GPU 信息
            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("⚠️  CUDA not available, using CPU")
        
        return True
    
    except ImportError:
        print("❌ PyTorch not installed")
        return False


def check_ollama():
    """检查 Ollama"""
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            encoding="utf-8",
        )
        
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ Ollama {version}")
            return True
        else:
            print("❌ Ollama not working")
            return False
    
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Ollama not found")
        return False


def check_nvidia_smi():
    """检查 nvidia-smi"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            encoding="utf-8"
        )
        
        if result.returncode == 0:
            print("✅ nvidia-smi available")
            return True
        else:
            print("❌ nvidia-smi not working")
            return False
    
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ nvidia-smi not found")
        return False


def main():
    """主检查函数"""
    print("TensorForge Dependency Checker")
    print("=" * 50)
    
    # 检查 Python 版本
    if not check_python_version():
        return False
    
    print("\nChecking core dependencies...")
    
    # 核心依赖
    core_deps = [
        ("torch", None, None),
        ("torchvision", None, None),
        ("numpy", None, None),
        ("requests", None, None),
        ("PyYAML", "yaml", None),
        ("loguru", None, None),
    ]
    
    missing_core = []
    for package, import_name, min_version in core_deps:
        if not check_package(package, import_name, min_version):
            missing_core.append(package)
    
    print("\nChecking ML dependencies...")
    
    # ML 依赖
    ml_deps = [
        ("transformers", None, None),
        ("diffusers", None, None),
        ("accelerate", None, None),
        ("ultralytics", None, None),
        ("faster_whisper", None, None),
        ("soundfile", None, None),
        ("huggingface_hub", None, None),
    ]
    
    missing_ml = []
    for package, import_name, min_version in ml_deps:
        if not check_package(package, import_name, min_version):
            missing_ml.append(package)
    
    print("\nChecking system dependencies...")
    
    # 系统依赖
    torch_ok = check_torch()
    ollama_ok = check_ollama()
    nvidia_ok = check_nvidia_smi()
    
    # 总结
    print("\n" + "=" * 50)
    print("Summary:")
    
    if missing_core:
        print(f"❌ Missing core dependencies: {', '.join(missing_core)}")
    else:
        print("✅ All core dependencies installed")
    
    if missing_ml:
        print(f"⚠️  Missing ML dependencies: {', '.join(missing_ml)}")
    else:
        print("✅ All ML dependencies installed")
    
    if not torch_ok:
        print("❌ PyTorch issues detected")
    if not ollama_ok:
        print("⚠️  Ollama not available (LLM tests will be skipped)")
    if not nvidia_ok:
        print("❌ nvidia-smi not available (GPU monitoring will fail)")
    
    # 安装建议
    if missing_core or missing_ml:
        print("\nInstallation commands:")
        
        if missing_core:
            print("# Core dependencies")
            print(f"pip install {' '.join(missing_core)}")
        
        if missing_ml:
            print("# ML dependencies")
            print(f"pip install {' '.join(missing_ml)}")
        
        print("\nOr install all at once:")
        all_missing = missing_core + missing_ml
        print(f"pip install {' '.join(all_missing)}")
    
    # PyTorch 安装建议
    if not torch_ok:
        print("\nPyTorch installation:")
        print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    return len(missing_core) == 0 and torch_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
