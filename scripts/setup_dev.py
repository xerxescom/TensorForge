#!/usr/bin/env python3
"""
开发环境设置脚本
"""
import subprocess
import sys
from pathlib import Path

def install_dependencies():
    """安装依赖"""
    print("📦 安装项目依赖...")
    
    requirements_files = [
        "requirements-core.txt",
        "requirements-ml.txt", 
        "requirements-dev.txt"
    ]
    
    for req_file in requirements_files:
        if Path(req_file).exists():
            print(f"   安装 {req_file}...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_file], 
                         check=False)

def setup_git_hooks():
    """设置 Git hooks"""
    print("🔧 设置 Git hooks...")
    
    hooks_dir = Path(".git/hooks")
    if hooks_dir.exists():
        # 预提交 hook - 代码格式检查
        pre_commit = hooks_dir / "pre-commit"
        pre_commit_content = """#!/bin/sh
# 预提交检查
python -m black --check src/ tests/
python -m isort --check-only src/ tests/
python -m mypy src/
"""
        pre_commit.write_text(pre_commit_content)
        pre_commit.chmod(0o755)
        print("   ✅ pre-commit hook 已设置")

def create_dev_config():
    """创建开发配置"""
    print("⚙️  创建开发配置...")
    
    from src.tensorforge.core.config_manager import config_manager
    config_manager.save_default_config()
    print("   ✅ 默认配置已生成")

def main():
    """主函数"""
    print("TensorForge 开发环境设置")
    print("=" * 40)
    
    try:
        install_dependencies()
        print()
        setup_git_hooks()
        print()
        create_dev_config()
        print()
        print("🎉 开发环境设置完成!")
        print("\n下一步:")
        print("1. 运行测试: python -m pytest")
        print("2. 运行示例: python examples/basic_usage.py")
        print("3. 开始开发: 编辑 src/ 目录下的文件")
        
    except Exception as e:
        print(f"❌ 设置失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
