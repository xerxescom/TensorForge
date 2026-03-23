"""
图像生成 / 视觉模型 / 语音模型 测试任务
支持平台：Windows 10/11 · Linux
"""
import time
import subprocess
import sys
import json
from pathlib import Path
from tf_logger import logger
from collector import BenchmarkRunner, _subprocess_kwargs
from model_manager import model_manager

IS_WINDOWS = sys.platform == "win32"


# ─────────────────────────────────────────────────────────────
#  图像生成 (Stable Diffusion / Flux via diffusers)
# ─────────────────────────────────────────────────────────────
class DiffusionBenchmark(BenchmarkRunner):
    """
    用 diffusers 本地跑图像生成，自动采集:
      - it/s (iterations per second)
      - seconds_per_image
      - steps_per_joule (由基类自动派生)
      - VRAM peak

    依赖: pip install diffusers accelerate torch

    使用示例:
        bench = DiffusionBenchmark(
            model_name="stabilityai/sdxl-turbo",
            precision="fp16",
            n_images=10,
            n_steps=20,
        )
        result = bench.run()
    """

    BENCH_PROMPTS = [
        "A futuristic city at night with neon lights reflecting on wet streets, cyberpunk style",
        "A serene mountain lake at sunrise, photorealistic, golden hour lighting",
        "A close-up portrait of a robot reading a book, detailed, cinematic",
        "Abstract geometric art, vivid colors, high contrast, minimalist",
        "A lush forest with rays of sunlight through the canopy, 8k",
    ]

    def __init__(
        self,
        model_name: str = "sdxl-turbo",
        precision: str = "fp16",
        n_images: int = 10,
        n_steps: int = 20,
        seed: int = 42,
        local_model_path: str = None,
        **kwargs,
    ):
        super().__init__(
            task_name="diffusion",
            model_name=model_name,
            precision=precision,
            **kwargs,
        )
        self.n_images = n_images
        self.n_steps = n_steps
        self.seed = seed
        self.local_model_path = local_model_path
        
        # 预下载模型
        self._ensure_model_available()

    def _ensure_model_available(self):
        """确保模型已下载"""
        if self.local_model_path:
            logger.info(f"[diffusion] Using local model: {self.local_model_path}")
            return
        
        try:
            logger.info(f"[diffusion] Pre-downloading model: {self.model_name}")
            model_path = model_manager.get_model_path(self.model_name)
            self.local_model_path = model_path
            logger.info(f"[diffusion] Model ready at: {model_path}")
        except Exception as e:
            logger.warning(f"[diffusion] Model download failed: {e}")
            logger.warning("[diffusion] Will try online loading during benchmark")

    def run_task(self) -> dict:
        logger.info(f"[diffusion] Starting diffusion benchmark: {self.n_images} images, {self.n_steps} steps each")
        
        # 用子进程运行，避免在同一 Python 进程中 OOM 时影响采集线程
        script = self._build_script()
        script_path = Path(self.output_dir) / "_diffusion_worker.py"
        script_path.write_text(script, "utf-8")
        logger.debug(f"[diffusion] Worker script written to: {script_path}")

        t0 = time.perf_counter()
        try:
            logger.debug(f"[diffusion] Starting worker subprocess with 15min timeout")
            out = subprocess.check_output(
                [sys.executable, str(script_path)],  # sys.executable = 当前 Python 路径，Windows/Linux 通用
                stderr=subprocess.STDOUT,
                timeout=900,  # 增加到15分钟
                **_subprocess_kwargs(),
            )
            elapsed = time.perf_counter() - t0
            logger.info(f"[diffusion] Worker completed in {elapsed:.2f}s")
            
            # Worker 最后一行输出 JSON 指标
            lines = out.decode().strip().splitlines()
            worker_metrics = json.loads(lines[-1])
            logger.debug(f"[diffusion] Worker metrics: {worker_metrics}")
        except subprocess.CalledProcessError as e:
            elapsed = time.perf_counter() - t0
            logger.warning(f"  [warn] Diffusion worker failed after {elapsed:.2f}s: {e.output.decode()[-500:]}")
            worker_metrics = {"error": "process_failed"}
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - t0
            logger.warning(f"  [warn] Diffusion worker timed out after {elapsed:.2f}s (15min limit)")
            worker_metrics = {"error": "timeout"}
        except (FileNotFoundError, json.JSONDecodeError) as e:
            elapsed = time.perf_counter() - t0
            logger.warning(f"  [warn] Diffusion worker error ({type(e).__name__}): {e}")
            worker_metrics = {"error": "dependency_missing"}

        # 计算性能指标
        total_steps = self.n_images * self.n_steps
        it_per_s = total_steps / elapsed if elapsed > 0 else 0
        seconds_per_image = elapsed / max(self.n_images, 1)
        
        logger.info(f"[diffusion] Results: {seconds_per_image:.2f}s per image, {it_per_s:.2f} iterations/s")
        
        return {
            "n_images": self.n_images,
            "n_steps": self.n_steps,
            "total_steps": total_steps,
            "total_elapsed_s": round(elapsed, 3),
            "seconds_per_image": round(seconds_per_image, 3),
            "it_per_s": round(it_per_s, 3),
            **worker_metrics,
        }

    def _build_script(self) -> str:
        """生成独立 Python worker 脚本"""
        dtype_map = {"fp16": "torch.float16", "fp32": "torch.float32", "bf16": "torch.bfloat16"}
        dtype = dtype_map.get(self.precision, "torch.float16")
        prompts_repr = repr(self.BENCH_PROMPTS)
        
        # 使用本地模型路径或在线模型名
        model_source = f'"{self.local_model_path}"' if self.local_model_path else f'"{self.model_name}"'
        
        return f"""
import torch, time, json, sys, os
from pathlib import Path

# 设置环境变量优化下载
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'  # 启用快速传输
os.environ['TRANSFORMERS_CACHE'] = '{model_manager.downloader.cache_dir}'
os.environ['HF_HOME'] = '{model_manager.downloader.cache_dir}'

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
            {model_source},
            torch_dtype={dtype},
            variant="fp16",
            local_files_only=True if {repr(self.local_model_path)} else False,
            resume_download=True,
            timeout=600,
        )
        print("[worker] Model loaded successfully")
    except Exception as e:
        print(f"[worker] Local model load failed: {{e}}")
        print("[worker] Trying online download...")
        
        pipe = AutoPipelineForText2Image.from_pretrained(
            {model_source},
            torch_dtype={dtype},
            variant="fp16",
            resume_download=True,
            timeout=600,
        )
    
    pipe = pipe.to("cuda")
    print("[worker] Model moved to GPU")

except ImportError as e:
    print(f"[worker] Import error: {{e}}")
    # 如果没有 diffusers，输出错误信息
    print(json.dumps({{"error": "diffusers_not_available", "details": str(e)}}))
    sys.exit(1)
except Exception as e:
    print(f"[worker] Model loading error: {{e}}")
    print(json.dumps({{"error": "model_load_failed", "details": str(e)}}))
    sys.exit(1)

prompts = {prompts_repr}
n = {self.n_images}
seed = {self.seed}
steps = {self.n_steps}
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
        print(f"[worker] Image {{i+1}}/{{n}} generated in {{elapsed:.3f}}s")
    except Exception as e:
        print(f"[worker] Image generation failed: {{e}}")
        # 仍然记录时间，但标记为失败
        times.append(time.perf_counter() - t0)

if times:
    result = {{
        "per_image_s_mean": round(sum(times)/len(times), 3),
        "per_image_s_min":  round(min(times), 3),
        "per_image_s_max":  round(max(times), 3),
        "success_count": success_count,
        "total_attempts": n,
        "success_rate": round(success_count / n, 4),
    }}
else:
    result = {{"error": "no_successful_generations"}}

print(json.dumps(result))
"""


# ─────────────────────────────────────────────────────────────
#  视觉模型 (YOLOv8 目标检测)
# ─────────────────────────────────────────────────────────────
class CVBenchmark(BenchmarkRunner):
    """
    用 ultralytics YOLOv8 跑目标检测，自动采集:
      - FPS (frames per second)
      - latency P50 / P95 / P99 (ms)
      - mAP (如提供 validation set)

    依赖: pip install ultralytics

    使用示例:
        bench = CVBenchmark(
            model_name="yolov8n",
            precision="fp16",
            n_frames=200,
            image_size=640,
        )
        result = bench.run()
    """

    def __init__(
        self,
        model_name: str = "yolov8n",
        precision: str = "fp16",
        n_frames: int = 200,
        image_size: int = 640,
        batch_size: int = 1,
        **kwargs,
    ):
        super().__init__(
            task_name="cv_detection",
            model_name=model_name,
            precision=precision,
            **kwargs,
        )
        self.n_frames = n_frames
        self.image_size = image_size
        self.batch_size = batch_size

    def run_task(self) -> dict:
        script = self._build_script()
        script_path = Path(self.output_dir) / "_cv_worker.py"
        script_path.write_text(script)

        t0 = time.perf_counter()
        try:
            out = subprocess.check_output(
                [sys.executable, str(script_path)],
                stderr=subprocess.STDOUT,
                timeout=300,
                **_subprocess_kwargs(),
            )
            elapsed = time.perf_counter() - t0
            lines = out.decode().strip().splitlines()
            worker_metrics = json.loads(lines[-1])
        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.warning(f"  [warn] CV worker failed: {e}")
            worker_metrics = {}

        return {
            "n_frames": self.n_frames,
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            "total_elapsed_s": round(elapsed, 3),
            **worker_metrics,
        }

    def _build_script(self) -> str:
        half = "True" if self.precision == "fp16" else "False"
        return f"""
import torch, time, json, numpy as np
from ultralytics import YOLO

model = YOLO("{self.model_name}.pt")
dummy = torch.zeros(1, 3, {self.image_size}, {self.image_size}).cuda()
half = {half}
if half:
    model.model.half()
    dummy = dummy.half()

# Warm-up
for _ in range(10):
    model(dummy, verbose=False)

latencies = []
for _ in range({self.n_frames}):
    t0 = time.perf_counter()
    model(dummy, verbose=False)
    latencies.append((time.perf_counter() - t0) * 1000)  # ms

lat = sorted(latencies)
n = len(lat)
print(json.dumps({{
    "fps":        round(1000 / (sum(latencies)/n), 2),
    "latency_p50_ms": round(lat[int(n*0.50)], 3),
    "latency_p95_ms": round(lat[int(n*0.95)], 3),
    "latency_p99_ms": round(lat[int(n*0.99)], 3),
    "latency_min_ms": round(lat[0], 3),
    "latency_max_ms": round(lat[-1], 3),
}}))
"""


# ─────────────────────────────────────────────────────────────
#  语音识别 (faster-whisper)
# ─────────────────────────────────────────────────────────────
class ASRBenchmark(BenchmarkRunner):
    """
    用 faster-whisper 跑语音识别，自动采集:
      - RTF (Real-Time Factor, <1 表示比实时快)
      - WER (Word Error Rate, 需提供 ground truth)
      - latency_ms

    依赖: pip install faster-whisper

    使用示例:
        bench = ASRBenchmark(
            model_name="large-v3",
            precision="float16",
            audio_files=["test1.wav", "test2.wav"],
            ground_truths=["hello world", "the quick brown fox"],
        )
        result = bench.run()
    """

    def __init__(
        self,
        model_name: str = "base",
        precision: str = "float16",
        audio_files: list[str] | None = None,
        ground_truths: list[str] | None = None,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(
            task_name="asr_whisper",
            model_name=model_name,
            precision=precision,
            **kwargs,
        )
        self.audio_files = audio_files or []
        self.ground_truths = ground_truths or []
        self.device = device

    def run_task(self) -> dict:
        if not self.audio_files:
            logger.warning("  [warn] No audio files provided, generating synthetic test")
            return self._synthetic_benchmark()

        script = self._build_script()
        script_path = Path(self.output_dir) / "_asr_worker.py"
        script_path.write_text(script)

        try:
            out = subprocess.check_output(
                [sys.executable, str(script_path)],
                stderr=subprocess.STDOUT,
                timeout=600,
                **_subprocess_kwargs(),
            )
            lines = out.decode().strip().splitlines()
            return json.loads(lines[-1])
        except Exception as e:
            logger.warning(f"  [warn] ASR worker failed: {e}")
            return {}

    def _synthetic_benchmark(self) -> dict:
        """无音频文件时，用合成数据测延迟和加载速度"""
        try:
            import numpy as np
            script = f"""
import time, json, numpy as np
from faster_whisper import WhisperModel

model = WhisperModel("{self.model_name}", device="{self.device}",
                     compute_type="{self.precision}")
# 合成 5 秒音频
audio = np.random.randn(5 * 16000).astype(np.float32)
latencies = []
for _ in range(5):
    t0 = time.perf_counter()
    segs, _ = model.transcribe(audio, language="en")
    list(segs)
    latencies.append(time.perf_counter() - t0)

mean_lat = sum(latencies) / len(latencies)
print(json.dumps({{
    "rtf": round(mean_lat / 5.0, 4),
    "latency_mean_s": round(mean_lat, 3),
    "audio_duration_s": 5.0,
    "note": "synthetic_audio_benchmark",
}}))
"""
            p = Path(self.output_dir) / "_asr_synth.py"
            p.write_text(script)
            out = subprocess.check_output(
                [sys.executable, str(p)],
                stderr=subprocess.STDOUT,
                timeout=120,
                **_subprocess_kwargs(),
            )
            lines = out.decode().strip().splitlines()
            return json.loads(lines[-1])
        except Exception as e:
            logger.warning(f"  [warn] Synthetic ASR failed: {e}")
            return {"note": "asr_not_available"}

    def _build_script(self) -> str:
        files_repr = repr(self.audio_files)
        truths_repr = repr(self.ground_truths)
        return f"""
import time, json
from faster_whisper import WhisperModel

model = WhisperModel("{self.model_name}", device="{self.device}",
                     compute_type="{self.precision}")

audio_files = {files_repr}
ground_truths = {truths_repr}

results = []
for i, fpath in enumerate(audio_files):
    import soundfile as sf
    audio, sr = sf.read(fpath)
    duration_s = len(audio) / sr

    t0 = time.perf_counter()
    segs, info = model.transcribe(fpath, language="en")
    transcript = " ".join(s.text for s in segs)
    elapsed = time.perf_counter() - t0

    rtf = round(elapsed / duration_s, 4) if duration_s > 0 else 0
    wer = None
    if i < len(ground_truths):
        ref = ground_truths[i].lower().split()
        hyp = transcript.lower().split()
        # 简单 WER 计算
        from difflib import SequenceMatcher
        sm = SequenceMatcher(None, ref, hyp)
        matches = sum(b.size for b in sm.get_matching_blocks())
        wer = round(1 - matches / max(len(ref), 1), 4)

    results.append({{
        "file": fpath,
        "rtf": rtf,
        "duration_s": round(duration_s, 2),
        "elapsed_s": round(elapsed, 3),
        "wer": wer,
    }})

rtf_list = [r["rtf"] for r in results]
wer_list = [r["wer"] for r in results if r["wer"] is not None]
print(json.dumps({{
    "n_files": len(results),
    "rtf_mean": round(sum(rtf_list)/len(rtf_list), 4) if rtf_list else 0,
    "rtf_min":  round(min(rtf_list), 4) if rtf_list else 0,
    "wer_mean": round(sum(wer_list)/len(wer_list), 4) if wer_list else None,
    "per_file_detail": results,
}}))
"""
