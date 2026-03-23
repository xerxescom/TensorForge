#!/usr/bin/env python3
"""
TensorForge 智能安装脚本
自动检测系统环境并安装合适的依赖
"""
import subprocess
import sys
import platform
from pathlib import Path


def check_python_version():
    """检查 Python 版本"""
    version = sys.version_info
    if version < (3, 8):
        print("❌ Python 3.8+ is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} OK")
        return True


def check_cuda():
    """检查 CUDA 版本"""
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            print(f"✅ CUDA {cuda_version} available")
            return cuda_version
        else:
            print("⚠️  CUDA not available, will use CPU version")
            return None
    except ImportError:
        print("⚠️  PyTorch not installed, checking system CUDA...")
        # 尝试从 nvidia-smi 检测
        try:
            result = subprocess.run(
                ["nvidia-smi"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0:
                # 解析 CUDA 版本
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'CUDA Version:' in line:
                        cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                        print(f"✅ System CUDA {cuda_version} detected")
                        return cuda_version
            return None
        except Exception:
            return None


def install_pytorch(cuda_version=None):
    """安装 PyTorch"""
    print("\n🚀 Installing PyTorch...")
    
    if cuda_version:
        # 选择合适的 CUDA 版本
        if cuda_version.startswith('12.'):
            index_url = "https://download.pytorch.org/whl/cu121"
        elif cuda_version.startswith('11.8'):
            index_url = "https://download.pytorch.org/whl/cu118"
        elif cuda_version.startswith('11.7'):
            index_url = "https://download.pytorch.org/whl/cu117"
        else:
            # 默认使用 CUDA 11.8
            index_url = "https://download.pytorch.org/whl/cu118"
        
        cmd = [
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", 
            f"--index-url={index_url}"
        ]
        print(f"Using CUDA index: {index_url}")
    else:
        # CPU 版本
        cmd = [
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", 
            "--index-url=https://download.pytorch.org/whl/cpu"
        ]
        print("Using CPU version")
    
    try:
        subprocess.check_call(cmd)
        print("✅ PyTorch installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ PyTorch installation failed: {e}")
        return False


def install_requirements():
    """安装其他依赖"""
    print("\n📦 Installing other dependencies...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ All dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Dependencies installation failed: {e}")
        return False


def check_ollama():
    """检查 Ollama"""
    try:
        result = subprocess.run(
            ["ollama", "version"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        if result.returncode == 0:
            print(f"✅ Ollama available: {result.stdout.strip()}")
            return True
        else:
            print("⚠️  Ollama not working")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("⚠️  Ollama not found")
        print("📥 Install Ollama from: https://ollama.com/download")
        return False


def check_nvidia_smi():
    """检查 nvidia-smi"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--version"], 
            capture_output=True, 
            text=True, 
            timeout=10
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
    """主安装函数"""
    print("🔧 TensorForge Installation Script")
    print("=" * 50)
    
    # 检查 Python 版本
    if not check_python_version():
        sys.exit(1)
    
    # 检查系统信息
    print(f"\n🖥️  System: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {sys.version}")
    
    # 检查 CUDA
    cuda_version = check_cuda()
    
    # 安装 PyTorch
    if not install_pytorch(cuda_version):
        print("❌ Failed to install PyTorch")
        sys.exit(1)
    
    # 安装其他依赖
    if not install_requirements():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # 检查系统工具
    print("\n🔍 Checking system tools...")
    nvidia_ok = check_nvidia_smi()
    ollama_ok = check_ollama()
    
    # 总结
    print("\n" + "=" * 50)
    print("📋 Installation Summary:")
    print(f"✅ PyTorch: Installed")
    print(f"✅ Dependencies: Installed")
    print(f"{'✅' if nvidia_ok else '❌'} nvidia-smi: {'Available' if nvidia_ok else 'Not available'}")
    print(f"{'✅' if ollama_ok else '⚠️'} Ollama: {'Available' if ollama_ok else 'Not available (optional)'}")
    
    if not nvidia_ok:
        print("\n⚠️  Warning: nvidia-smi not available")
        print("   GPU monitoring will not work")
        print("   Please install NVIDIA drivers")
    
    if not ollama_ok:
        print("\n⚠️  Ollama not available (optional)")
        print("   LLM benchmarks will be skipped")
        print("   Install from: https://ollama.com/download")
    
    print("\n🎉 Installation completed!")
    print("\nNext steps:")
    print("1. Run dependency check: python check_deps.py")
    print("2. Test model download: python test_models.py")
    print("3. Run benchmark: python run_suite.py")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Installation failed: {e}")
        sys.exit(1)
