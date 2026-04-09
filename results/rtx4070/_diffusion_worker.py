
import json
import os
import sys
import time

import torch

# 设置环境变量优化下载
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'  # 启用快速传输
os.environ['TRANSFORMERS_CACHE'] = 'models_cache'
os.environ['HF_HOME'] = 'models_cache'

# 网络配置
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    from diffusers import AutoPipelineForText2Image
    from huggingface_hub import hf_hub_download, snapshot_download

    print("[worker] Loading diffusion model...")

    # 尝试加载本地模型
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            r"models_cache\stabilityai_sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            local_files_only=True if 'models_cache\\stabilityai_sdxl-turbo' else False,
            resume_download=True,
            timeout=600,
        )
        print("[worker] Model loaded successfully")
    except Exception as e:
        print(f"[worker] Local model load failed: {e}")
        print("[worker] Trying online download...")

        pipe = AutoPipelineForText2Image.from_pretrained(
            r"models_cache\stabilityai_sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            resume_download=True,
            timeout=600,
        )

    pipe = pipe.to("cuda")
    print("[worker] Model moved to GPU")

except ImportError as e:
    print(f"[worker] Import error: {e}")
    # 如果没有 diffusers，输出错误信息
    print(json.dumps({"error": "diffusers_not_available", "details": str(e)}))
    sys.exit(1)
except Exception as e:
    print(f"[worker] Model loading error: {e}")
    print(json.dumps({"error": "model_load_failed", "details": str(e)}))
    sys.exit(1)

prompts = ['A futuristic city at night with neon lights reflecting on wet streets, cyberpunk style', 'A serene mountain lake at sunrise, photorealistic, golden hour lighting', 'A close-up portrait of a robot reading a book, detailed, cinematic', 'Abstract geometric art, vivid colors, high contrast, minimalist', 'A lush forest with rays of sunlight through the canopy, 8k']
n = 10
seed = 42
steps = 20
generator = torch.Generator("cuda").manual_seed(seed)

times = []
success_count = 0

for i in range(n):
    prompt = prompts[i % len(prompts)]
    t0 = time.perf_counter()

    try:
        image = pipe(prompt=prompt, num_inference_steps=steps, generator=generator).images[0]
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        success_count += 1
        print(f"[worker] Image {i+1}/{n} generated in {elapsed:.3f}s")
    except Exception as e:
        print(f"[worker] Image generation failed: {e}")
        # 仍然记录时间，但标记为失败
        times.append(time.perf_counter() - t0)

if times:
    result = {
        "per_image_s_mean": round(sum(times)/len(times), 3),
        "per_image_s_min":  round(min(times), 3),
        "per_image_s_max":  round(max(times), 3),
        "success_count": success_count,
        "total_attempts": n,
        "success_rate": round(success_count / n, 4),
    }
else:
    result = {"error": "no_successful_generations"}

print(json.dumps(result))
