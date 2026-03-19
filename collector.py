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
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

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
    print(f"[warn] {_e}")


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

    # 性能指标 (由子类填充)
    metrics: dict = field(default_factory=dict)

    # 自动采集的 GPU 统计（统计值）
    gpu_stats: dict = field(default_factory=dict)

    # 原始采样序列（可选，用于画时序图）
    raw_samples: list = field(default_factory=list)


class GPUSampler:
    """
    后台线程，以固定间隔轮询 nvidia-smi，
    线程安全地存储样本列表。
    """

    def __init__(self, interval_s: float = 0.5, gpu_index: int = 0):
        self.interval_s = interval_s
        self.gpu_index = gpu_index
        self._samples: list[GPUSample] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    #  nvidia-smi 查询                                                    #
    # ------------------------------------------------------------------ #
    def _query(self) -> Optional[GPUSample]:
        if NVIDIA_SMI is None:
            return None

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

            return GPUSample(
                timestamp=time.time(),
                gpu_util=safe(vals[0]),
                mem_used_mb=safe(vals[1]),
                mem_total_mb=safe(vals[2]),
                power_w=safe(vals[3]),
                temp_c=safe(vals[4]),
                sm_clock_mhz=safe(vals[5]),
                mem_clock_mhz=safe(vals[6]),
            )
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    #  生命周期                                                           #
    # ------------------------------------------------------------------ #
    def start(self):
        self._stop_event.clear()
        with self._lock:
            self._samples.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[GPUSample]:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        with self._lock:
            return list(self._samples)

    def _loop(self):
        while not self._stop_event.is_set():
            sample = self._query()
            if sample:
                with self._lock:
                    self._samples.append(sample)
            self._stop_event.wait(self.interval_s)

    # ------------------------------------------------------------------ #
    #  统计摘要                                                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def summarize(samples: list[GPUSample]) -> dict:
        if not samples:
            return {}

        def _stats(values):
            if not values:
                return {}
            s = sorted(values)
            n = len(s)
            return {
                "mean": round(sum(s) / n, 2),
                "max": round(s[-1], 2),
                "min": round(s[0], 2),
                "p95": round(s[int(n * 0.95)], 2),
            }

        return {
            "gpu_util_%":    _stats([s.gpu_util for s in samples]),
            "power_w":       _stats([s.power_w for s in samples]),
            "temp_c":        _stats([s.temp_c for s in samples]),
            "mem_used_mb":   _stats([s.mem_used_mb for s in samples]),
            "sm_clock_mhz":  _stats([s.sm_clock_mhz for s in samples]),
            "sample_count":  len(samples),
        }


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
        print(f"[{self.task_name}] Warm-up {self.warmup_s}s ...")
        time.sleep(self.warmup_s)

        print(f"[{self.task_name}] Starting GPU sampler ...")
        self._sampler.start()
        start_ts = time.time()
        start_str = datetime.now().isoformat(timespec="seconds")

        print(f"[{self.task_name}] Running task ...")
        metrics = self.run_task()

        end_ts = time.time()
        end_str = datetime.now().isoformat(timespec="seconds")
        samples = self._sampler.stop()

        # 派生功耗效率指标
        duration = round(end_ts - start_ts, 3)
        gpu_stats = GPUSampler.summarize(samples)
        metrics = self._enrich_efficiency(metrics, gpu_stats, duration)

        result = BenchmarkResult(
            task_name=self.task_name,
            model_name=self.model_name,
            precision=self.precision,
            start_time=start_str,
            end_time=end_str,
            duration_s=duration,
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
    def _enrich_efficiency(self, metrics: dict, gpu_stats: dict, duration_s: float) -> dict:
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
            "start_time": result.start_time,
            "duration_s": result.duration_s,
            **{f"metric_{k}": v for k, v in result.metrics.items()},
            "gpu_util_mean": result.gpu_stats.get("gpu_util_%", {}).get("mean"),
            "power_mean_w": result.gpu_stats.get("power_w", {}).get("mean"),
            "power_max_w": result.gpu_stats.get("power_w", {}).get("max"),
            "temp_max_c": result.gpu_stats.get("temp_c", {}).get("max"),
            "mem_used_max_mb": result.gpu_stats.get("mem_used_mb", {}).get("max"),
        }
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        print(f"[{self.task_name}] Saved → {json_path.name}  +  summary.csv")

    def _print_summary(self, r: BenchmarkResult):
        print(f"\n{'='*52}")
        print(f"  {r.task_name} | {r.model_name} | {r.precision}")
        print(f"  Duration : {r.duration_s}s")
        print(f"  Metrics  : {json.dumps(r.metrics, indent=4)}")
        pw = r.gpu_stats.get("power_w", {})
        tc = r.gpu_stats.get("temp_c", {})
        mu = r.gpu_stats.get("mem_used_mb", {})
        print(f"  Power    : mean={pw.get('mean')}W  max={pw.get('max')}W")
        print(f"  Temp     : mean={tc.get('mean')}°C  max={tc.get('max')}°C")
        print(f"  VRAM     : max={mu.get('max')}MB")
        print(f"{'='*52}\n")
