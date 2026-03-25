import os
import tarfile
from datetime import datetime

EXCLUDE_DIRS = {".idea", ".git", "__pycache__"}

def should_exclude(path):
    parts = set(path.split(os.sep))
    return not EXCLUDE_DIRS.isdisjoint(parts)

def make_tar(source_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"project_{timestamp}.tar"

    with tarfile.open(output_name, "w") as tar:
        for root, dirs, files in os.walk(source_dir):
            # 过滤目录
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for file in files:
                full_path = os.path.join(root, file)

                if should_exclude(full_path):
                    continue

                arcname = os.path.relpath(full_path, source_dir)
                tar.add(full_path, arcname=arcname)

    print(f"✅ 打包完成: {output_name}")


if __name__ == "__main__":
    make_tar(".")