"""
GPU Benchmark - Automated Data Collector
后台持续采样 GPU 状态，与任意测试任务解耦
支持平台：Windows 10/11 · Linux
"""
import threading
import time
import json
import csv
import subprocess
import sys
import platform
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from tf_logger import logger

# ── 平台工具 ────────────────────────────────────────────────────────────────

IS_WINDOWS = sys.platform == "win32"

if not IS_WINDOWS and sys.platform != "linux":
    raise SystemError(f"Unsupported platform: {sys.platform}. Only Windows and Linux are supported.")


def _find_nvidia_smi() -> str:
    """
    返回 nvidia-smi 的可执行路径。
    Linux 上直接用 PATH 里的即可；
    Windows 上 nvidia-smi.exe 有时不在 PATH，需要主动搜索驱动目录。
    """
    if IS_WINDOWS:
        # 先试 PATH
        try:
            subprocess.check_output(
                ["nvidia-smi", "--version"],
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            return "nvidia-smi"
        except Exception:
            pass

        # Windows 驱动常见安装路径
        candidates = [
            r"C:\Windows\System32\nvidia-smi.exe",
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        ]
        # DriverStore：Win10/11 驱动包存储位置
        driver_store = Path(r"C:\Windows\System32\DriverStore\FileRepository")
        if driver_store.exists():
            for p in driver_store.glob("nv_dispi.inf_amd64_*/nvidia-smi.exe"):
                candidates.append(str(p))

        for c in candidates:
            if Path(c).exists():
                return c

        raise FileNotFoundError(
            "nvidia-smi not found. Make sure NVIDIA drivers are installed.\n"
            "Check: C:\\Windows\\System32\\ or "
            "C:\\Program Files\\NVIDIA Corporation\\NVSMI\\"
        )

    else:  # Linux
        return "nvidia-smi"


def _subprocess_kwargs() -> dict:
    """
    Windows 上加 CREATE_NO_WINDOW，避免每次 nvidia-smi 调用弹出黑色 cmd 窗口。
    Linux 不需要该 flag。
    """
    if IS_WINDOWS:
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


# 缓存一次，避免每次采样都重新搜索
try:
    NVIDIA_SMI = _find_nvidia_smi()
except FileNotFoundError as _e:
    NVIDIA_SMI = None
    logger.warning(_e)


@dataclass
class GPUSample:
    timestamp: float
    gpu_util: float        # %
    mem_used_mb: float
    mem_total_mb: float
    power_w: float
    temp_c: float
    sm_clock_mhz: float
    mem_clock_mhz: float


@dataclass
class BenchmarkResult:
    task_name: str
    model_name: str
    precision: str         # fp32 / fp16 / int8 / int4
    start_time: str
    end_time: str
    duration_s: float
    status: str = "success"
    error: Optional[str] = None
    environment: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    # 性能指标 (由子类填充)
    metrics: dict = field(default_factory=dict)

    # 自动采集的 GPU 统计（统计值）
    gpu_stats: dict = field(default_factory=dict)

    # 原始采样序列（可选，用于画时序图）
    raw_samples: list = field(default_factory=list)


class GPUSampler:
    """
    后台线程，以动态间隔轮询 nvidia-smi，
    线程安全地存储样本列表。
    """

    def __init__(self, interval_s: float = 0.5, gpu_index: int = 0, adaptive_sampling: bool = True):
        self.interval_s = interval_s
        self.base_interval_s = interval_s
        self.gpu_index = gpu_index
        self.adaptive_sampling = adaptive_sampling
        self._samples: list[GPUSample] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._query_count = 0
        self._error_count = 0
        self._timeout_count = 0
        self._query_latency_ms: list[float] = []
        self._last_error: Optional[str] = None
        self._last_gpu_util = 0.0

    # ------------------------------------------------------------------ #
    #  nvidia-smi 查询                                                    #
    # ------------------------------------------------------------------ #
    def _query(self) -> tuple[Optional[GPUSample], Optional[str], float]:
        if NVIDIA_SMI is None:
            return None, "nvidia-smi unavailable", 0.0

        fields = (
            "utilization.gpu,"
            "memory.used,"
            "memory.total,"
            "power.draw,"
            "temperature.gpu,"
            "clocks.current.sm,"
            "clocks.current.memory"
        )
        cmd = [
            NVIDIA_SMI,
            f"--id={self.gpu_index}",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ]
        t0 = time.perf_counter()
        try:
            out = subprocess.check_output(
                cmd,
                stderr=subprocess.DEVNULL,
                timeout=3,
                **_subprocess_kwargs(),
            )
            vals = [v.strip() for v in out.decode().strip().split(",")]

            def safe(v, default=0.0):
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return default

            latency_ms = (time.perf_counter() - t0) * 1000
            return GPUSample(
                timestamp=time.time(),
                gpu_util=safe(vals[0]),
                mem_used_mb=safe(vals[1]),
                mem_total_mb=safe(vals[2]),
                power_w=safe(vals[3]),
                temp_c=safe(vals[4]),
                sm_clock_mhz=safe(vals[5]),
                mem_clock_mhz=safe(vals[6]),
            ), None, latency_ms
        except subprocess.TimeoutExpired:
            latency_ms = (time.perf_counter() - t0) * 1000
            return None, "nvidia-smi query timeout", latency_ms
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            return None, str(e), latency_ms

    # ------------------------------------------------------------------ #
    #  生命周期                                                           #
    # ------------------------------------------------------------------ #
    def start(self):
        self._stop_event.clear()
        with self._lock:
            self._samples.clear()
        self._query_count = 0
        self._error_count = 0
        self._timeout_count = 0
        self._query_latency_ms.clear()
        self._last_error = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def smart_sampling(self, samples: list[GPUSample], max_samples: int = 1000) -> list[GPUSample]:
        """智能采样：保留关键数据点，减少内存使用"""
        if len(samples) <= max_samples:
            return samples
        
        # 保留首尾样本
        result = [samples[0], samples[-1]]
        
        # 找到峰值和谷值
        gpu_utils = [s.gpu_util for s in samples]
        power_vals = [s.power_w for s in samples]
        temp_vals = [s.temp_c for s in samples]
        
        # 找出关键点索引
        peak_indices = set()
        for values in [gpu_utils, power_vals, temp_vals]:
            peak_idx = values.index(max(values))
            valley_idx = values.index(min(values))
            peak_indices.add(peak_idx)
            peak_indices.add(valley_idx)
        
        # 均匀采样中间点
        remaining_slots = max_samples - len(result) - len(peak_indices)
        if remaining_slots > 0:
            step = len(samples) // remaining_slots
            uniform_indices = set(range(0, len(samples), step))
        else:
            uniform_indices = set()
        
        # 合并所有索引并排序
        all_indices = (peak_indices | uniform_indices) - {0, len(samples) - 1}
        for idx in sorted(all_indices):
            if len(result) < max_samples:
                result.append(samples[idx])
        
        return sorted(result, key=lambda x: x.timestamp)

    def stop(self, max_raw_samples: int = 1000) -> list[GPUSample]:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        with self._lock:
            samples = list(self._samples)
            return self.smart_sampling(samples, max_raw_samples)

    def _adaptive_interval(self, current_util: float) -> float:
        """根据 GPU 利用率动态调整采样间隔"""
        if not self.adaptive_sampling:
            return self.base_interval_s
        
        if current_util > 80.0:
            return self.base_interval_s * 0.4  # 高负载时更频繁采样
        elif current_util < 20.0:
            return self.base_interval_s * 2.0  # 低负载时降低频率
        else:
            return self.base_interval_s

    def _loop(self):
        while not self._stop_event.is_set():
            sample, error, latency_ms = self._query()
            self._query_count += 1
            self._query_latency_ms.append(round(latency_ms, 3))
            if sample:
                with self._lock:
                    self._samples.append(sample)
                    self._last_gpu_util = sample.gpu_util
                # 动态调整下次采样间隔
                next_interval = self._adaptive_interval(sample.gpu_util)
            else:
                self._error_count += 1
                if error == "nvidia-smi query timeout":
                    self._timeout_count += 1
                self._last_error = error
                next_interval = self.base_interval_s
            
            self._stop_event.wait(next_interval)

    # ------------------------------------------------------------------ #
    #  统计摘要                                                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def summarize(samples: list[GPUSample]) -> dict:
        def _stats(values):
            if not values:
                return {}
            s = sorted(values)
            n = len(s)
            p95_idx = min(max(int(n * 0.95), 0), n - 1)
            return {
                "mean": round(sum(s) / n, 2),
                "max": round(s[-1], 2),
                "min": round(s[0], 2),
                "p95": round(s[p95_idx], 2),
            }

        if not samples:
            return {"sample_count": 0}

        return {
            "gpu_util_%":    _stats([s.gpu_util for s in samples]),
            "power_w":       _stats([s.power_w for s in samples]),
            "temp_c":        _stats([s.temp_c for s in samples]),
            "mem_used_mb":   _stats([s.mem_used_mb for s in samples]),
            "sm_clock_mhz":  _stats([s.sm_clock_mhz for s in samples]),
            "sample_count":  len(samples),
        }

    def health(self) -> dict:
        avg_latency = (
            round(sum(self._query_latency_ms) / len(self._query_latency_ms), 3)
            if self._query_latency_ms else 0.0
        )
        drop_rate = (
            round(self._error_count / self._query_count, 4)
            if self._query_count else 0.0
        )
        return {
            "query_count": self._query_count,
            "success_count": max(self._query_count - self._error_count, 0),
            "error_count": self._error_count,
            "timeout_count": self._timeout_count,
            "drop_rate": drop_rate,
            "avg_query_latency_ms": avg_latency,
            "last_error": self._last_error,
        }


def _enrich_efficiency(metrics: dict, gpu_stats: dict, duration_s: float) -> dict:
    """自动计算功耗效率指标"""
    mean_power = gpu_stats.get("power_w", {}).get("mean", 0)
    if mean_power > 0 and duration_s > 0:
        energy_j = mean_power * duration_s
        metrics["energy_j"] = round(energy_j, 2)

        # tokens per joule (LLM)
        if "tokens_generated" in metrics:
            metrics["tokens_per_joule"] = round(
                metrics["tokens_generated"] / energy_j, 4
            )
        # iterations per joule (Diffusion)
        if "total_steps" in metrics:
            metrics["steps_per_joule"] = round(
                metrics["total_steps"] / energy_j, 4
            )
    return metrics


class BenchmarkRunner:
    """
    所有测试任务的基类。
    子类只需实现 run_task() → dict (业务指标)
    其余采集、计时、保存全部自动完成。
    """

    def __init__(
        self,
        task_name: str,
        model_name: str,
        precision: str = "fp16",
        output_dir: str = "results",
        gpu_index: int = 0,
        sample_interval_s: float = 0.5,
        warmup_s: float = 5.0,
        keep_raw_samples: bool = True,
    ):
        self.task_name = task_name
        self.model_name = model_name
        self.precision = precision
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_index = gpu_index
        self.sample_interval_s = sample_interval_s
        self.warmup_s = warmup_s
        self.keep_raw_samples = keep_raw_samples
        self._sampler = GPUSampler(sample_interval_s, gpu_index)

    # ------------------------------------------------------------------ #
    #  子类必须实现                                                       #
    # ------------------------------------------------------------------ #
    def run_task(self) -> dict:
        """执行实际推理/生成任务，返回业务指标 dict"""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    #  主流程 (自动化)                                                   #
    # ------------------------------------------------------------------ #
    def run(self) -> BenchmarkResult:
        logger.info(f"[{self.task_name}] Warm-up {self.warmup_s}s ...")
        time.sleep(self.warmup_s)

        logger.info(f"[{self.task_name}] Starting GPU sampler ...")
        self._sampler.start()
        start_ts = time.time()
        start_str = datetime.now().isoformat(timespec="seconds")
        metrics = {}
        status = "success"
        error = None

        logger.info(f"[{self.task_name}] Running task ...")
        try:
            metrics = self.run_task()
        except Exception as e:
            status = "failed"
            error = str(e)
            logger.exception(f"[{self.task_name}] [error] {error}")

        end_ts = time.time()
        end_str = datetime.now().isoformat(timespec="seconds")
        samples = self._sampler.stop()

        # 派生功耗效率指标
        duration = round(end_ts - start_ts, 3)
        gpu_stats = GPUSampler.summarize(samples)
        gpu_stats["sampler_health"] = self._sampler.health()
        metrics = _enrich_efficiency(metrics, gpu_stats, duration)

        result = BenchmarkResult(
            task_name=self.task_name,
            model_name=self.model_name,
            precision=self.precision,
            start_time=start_str,
            end_time=end_str,
            duration_s=duration,
            status=status,
            error=error,
            environment=self.collect_environment(),
            config=self.build_config_snapshot(),
            metrics=metrics,
            gpu_stats=gpu_stats,
            raw_samples=(
                [asdict(s) for s in samples] if self.keep_raw_samples else []
            ),
        )

        self._save(result)
        self._print_summary(result)
        return result

    # ------------------------------------------------------------------ #
    #  内部工具                                                           #
    # ------------------------------------------------------------------ #

    def collect_environment(self) -> dict:
        env = {
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "gpu_index": self.gpu_index,
            "sample_interval_s": self.sample_interval_s,
            "nvidia_smi": NVIDIA_SMI,
        }
        if NVIDIA_SMI:
            try:
                out = subprocess.check_output(
                    [
                        NVIDIA_SMI,
                        f"--id={self.gpu_index}",
                        "--query-gpu=name,driver_version,memory.total,power.limit",
                        "--format=csv,noheader,nounits",
                    ],
                    stderr=subprocess.DEVNULL,
                    timeout=3,
                    **_subprocess_kwargs(),
                ).decode().strip()
                vals = [v.strip() for v in out.split(",")]
                if len(vals) >= 4:
                    env["gpu"] = {
                        "name": vals[0],
                        "driver_version": vals[1],
                        "memory_total_mb": vals[2],
                        "power_limit_w": vals[3],
                    }
            except Exception as e:
                env["gpu_probe_error"] = str(e)
        return env

    def build_config_snapshot(self) -> dict:
        return {
            "task_name": self.task_name,
            "model_name": self.model_name,
            "precision": self.precision,
            "output_dir": str(self.output_dir),
            "gpu_index": self.gpu_index,
            "sample_interval_s": self.sample_interval_s,
            "warmup_s": self.warmup_s,
            "keep_raw_samples": self.keep_raw_samples,
        }

    @staticmethod
    def _flatten_scalar_metrics(metrics: dict) -> dict:
        out = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                out[f"metric_{k}"] = v
        return out

    def _save(self, result: BenchmarkResult):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{self.task_name}_{self.model_name}_{self.precision}_{ts}"

        # JSON (完整数据)
        json_path = self.output_dir / f"{stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2, ensure_ascii=False)

        # CSV (指标摘要，方便 Excel / pandas 对比)
        csv_path = self.output_dir / "summary.csv"
        row = {
            "task": result.task_name,
            "model": result.model_name,
            "precision": result.precision,
            "status": result.status,
            "start_time": result.start_time,
            "duration_s": result.duration_s,
            **self._flatten_scalar_metrics(result.metrics),
            "gpu_util_mean": result.gpu_stats.get("gpu_util_%", {}).get("mean"),
            "power_mean_w": result.gpu_stats.get("power_w", {}).get("mean"),
            "power_max_w": result.gpu_stats.get("power_w", {}).get("max"),
            "temp_max_c": result.gpu_stats.get("temp_c", {}).get("max"),
            "mem_used_max_mb": result.gpu_stats.get("mem_used_mb", {}).get("max"),
            "sampler_drop_rate": result.gpu_stats.get("sampler_health", {}).get("drop_rate"),
        }
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        logger.info(f"[{self.task_name}] Saved → {json_path.name}  +  summary.csv")

    def _print_summary(self, r: BenchmarkResult):
        logger.info("=" * 52)
        logger.info(f"  {r.task_name} | {r.model_name} | {r.precision}")
        logger.info(f"  Duration : {r.duration_s}s")
        logger.info(f"  Metrics  : {json.dumps(r.metrics, indent=4)}")
        pw = r.gpu_stats.get("power_w", {})
        tc = r.gpu_stats.get("temp_c", {})
        mu = r.gpu_stats.get("mem_used_mb", {})
        logger.info(f"  Power    : mean={pw.get('mean')}W  max={pw.get('max')}W")
        logger.info(f"  Temp     : mean={tc.get('mean')}°C  max={tc.get('max')}°C")
        logger.info(f"  VRAM     : max={mu.get('max')}MB")
        logger.info("=" * 52)
