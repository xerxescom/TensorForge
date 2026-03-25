"""
GPU Benchmark - Automated Data Collector
后台持续采样 GPU 状态，与任意测试任务解耦
支持平台：Windows 10/11 · Linux
"""
import csv
import json
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

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
            for item in driver_store.iterdir():
                if item.is_dir() and "nv_dispi" in item.name.lower():
                    nvidia_smi_path = item / "nvidia-smi.exe"
                    if nvidia_smi_path.exists():
                        candidates.append(str(nvidia_smi_path))
                        break

        for candidate in candidates:
            try:
                subprocess.check_output(
                    [candidate, "--version"],
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    timeout=5,
                )
                return candidate
            except Exception:
                continue

        raise FileNotFoundError(
            "nvidia-smi.exe not found. Please install NVIDIA drivers or add nvidia-smi to PATH."
        )
    else:
        # Linux 直接用 PATH
        return "nvidia-smi"


def _subprocess_kwargs() -> Dict[str, Any]:
    """返回平台特定的 subprocess 参数"""
    if IS_WINDOWS:
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    else:
        return {}


@dataclass
class GPUSample:
    """GPU 样本数据结构"""
    timestamp: float
    gpu_util: float          # GPU 利用率 (%)
    memory_used_mb: float    # 显存使用 (MB)
    memory_total_mb: float   # 显存总量 (MB)
    power_w: float          # 功耗 (W)
    temp_c: float           # 温度 (°C)
    clock_mhz: float        # 核心频率 (MHz)
    memory_clock_mhz: float  # 显存频率 (MHz)


@dataclass
class BenchmarkResult:
    """基准测试结果数据结构"""
    task_name: str
    model_name: str
    precision: str
    status: str  # "ok", "error", "timeout"
    duration_s: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    gpu_stats: Dict[str, Any] = field(default_factory=dict)
    raw_samples: List[GPUSample] = field(default_factory=list)
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


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
        self._samples: List[GPUSample] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._query_count = 0
        self._error_count = 0
        self._timeout_count = 0
        self._query_latency_ms: List[float] = []
        self._last_error: Optional[str] = None
        self._last_gpu_util = 0.0

    def start(self) -> None:
        """启动后台采样线程"""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, max_raw_samples: int = 1000) -> List[GPUSample]:
        """停止采样并返回样本列表"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        with self._lock:
            samples = list(self._samples)
            return self.smart_sampling(samples, max_raw_samples)

    def smart_sampling(self, samples: List[GPUSample], max_samples: int = 1000) -> List[GPUSample]:
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
            if values:
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

    def _query(self) -> tuple[Optional[GPUSample], Optional[str], float]:
        """查询一次 nvidia-smi，返回 (样本, 错误, 延迟ms)"""
        start = time.perf_counter()
        try:
            cmd = [
                _find_nvidia_smi(),
                "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu,clock.sm,clock.memory",
                "--format=csv,noheader,nounits",
                f"--id={self.gpu_index}",
            ]
            out = subprocess.check_output(
                cmd,
                stderr=subprocess.DEVNULL,
                timeout=5,
                **_subprocess_kwargs(),
            )
            latency = (time.perf_counter() - start) * 1000
            parts = out.decode().strip().split(", ")
            if len(parts) != 7:
                return None, f"Unexpected output: {out.decode()}", latency
            
            gpu_util = float(parts[0])
            memory_used = float(parts[1])
            memory_total = float(parts[2])
            power = float(parts[3]) if parts[3] != "[N/A]" else 0.0
            temp = float(parts[4]) if parts[4] != "[N/A]" else 0.0
            clock_sm = float(parts[5])
            clock_mem = float(parts[6])
            
            sample = GPUSample(
                timestamp=time.time(),
                gpu_util=gpu_util,
                memory_used_mb=memory_used,
                memory_total_mb=memory_total,
                power_w=power,
                temp_c=temp,
                clock_mhz=clock_sm,
                memory_clock_mhz=clock_mem,
            )
            return sample, None, latency
        except subprocess.TimeoutExpired:
            return None, "nvidia-smi query timeout", (time.perf_counter() - start) * 1000
        except subprocess.CalledProcessError as e:
            return None, f"nvidia-smi query failed: {e}", (time.perf_counter() - start) * 1000
        except Exception as e:
            return None, f"Unexpected error: {e}", (time.perf_counter() - start) * 1000

    def _loop(self):
        """后台采样循环"""
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

    def get_stats(self) -> Dict[str, Any]:
        """返回采样器统计信息"""
        with self._lock:
            if not self._samples:
                return {
                    "query_count": self._query_count,
                    "error_count": self._error_count,
                    "timeout_count": self._timeout_count,
                    "error_rate": self._error_count / max(self._query_count, 1),
                    "avg_latency_ms": 0,
                    "last_error": self._last_error,
                    "mean_util": 0.0,
                    "mean_power": 0.0,
                }
            
            gpu_utils = [s.gpu_util for s in self._samples]
            power_vals = [s.power_w for s in self._samples if s.power_w > 0]
            temps = [s.temp_c for s in self._samples if s.temp_c > 0]
            
            stats = {
                "query_count": self._query_count,
                "error_count": self._error_count,
                "timeout_count": self._timeout_count,
                "error_rate": self._error_count / max(self._query_count, 1),
                "avg_latency_ms": round(sum(self._query_latency_ms) / len(self._query_latency_ms), 3) if self._query_latency_ms else 0,
                "last_error": self._last_error,
                "sample_count": len(self._samples),
                "gpu_util_mean": round(statistics.mean(gpu_utils), 2) if gpu_utils else 0,
                "gpu_util_max": max(gpu_utils) if gpu_utils else 0,
                "gpu_util_min": min(gpu_utils) if gpu_utils else 0,
                "power_mean": round(statistics.mean(power_vals), 2) if power_vals else 0,
                "power_max": max(power_vals) if power_vals else 0,
                "temp_mean": round(statistics.mean(temps), 2) if temps else 0,
                "temp_max": max(temps) if temps else 0,
            }
            # 向后兼容旧字段名
            stats["mean_util"] = stats["gpu_util_mean"]
            stats["mean_power"] = stats["power_mean"]
            return stats


class BenchmarkRunner:
    """
    基准测试运行器基类。
    子类需要实现 run_task() 方法，返回业务指标字典。
    """

    def __init__(
        self,
        task_name: str,
        model_name: str,
        precision: str = "fp16",
        gpu_index: int = 0,
        warmup_s: float = 5.0,
        timeout_s: int = 120,
        output_dir: str = "results",
        **kwargs,
    ):
        self.task_name = task_name
        self.model_name = model_name
        self.precision = precision
        self.gpu_index = gpu_index
        self.warmup_s = warmup_s
        self.timeout_s = timeout_s
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sampler = GPUSampler(gpu_index=gpu_index)

    def run(self) -> BenchmarkResult:
        """完整运行一次基准测试，包括预热、采样、结果保存"""
        print(f"Starting {self.task_name} benchmark...")
        self.sampler.start()
        
        # 预热
        if self.warmup_s > 0:
            print(f"  Warming up {self.warmup_s}s...")
            time.sleep(self.warmup_s)
        
        # 运行实际任务
        start_time = time.perf_counter()
        try:
            metrics = self.run_task()
            status = "ok"
            error = None
        except subprocess.TimeoutExpired:
            metrics = {}
            status = "timeout"
            error = "Task timeout"
        except Exception as e:
            metrics = {}
            status = "error"
            error = str(e)
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        # 停止采样
        samples = self.sampler.stop()
        
        # 计算GPU统计
        gpu_stats = self._calc_gpu_stats(samples)
        
        # 构建结果
        result = BenchmarkResult(
            task_name=self.task_name,
            model_name=self.model_name,
            precision=self.precision,
            status=status,
            duration_s=round(duration, 3),
            metrics=metrics,
            gpu_stats=gpu_stats,
            raw_samples=samples,
            error=error,
        )
        
        # 保存结果
        self._save_result(result)
        self._append_summary_csv(result)
        
        print(f"  Completed in {duration:.2f}s, status: {status}")
        return result

    def run_task(self) -> Dict[str, Any]:
        """子类需要实现的业务逻辑"""
        raise NotImplementedError("Subclass must implement run_task()")

    def _calc_gpu_stats(self, samples: List[GPUSample]) -> Dict[str, Any]:
        """从原始采样计算GPU统计指标"""
        if not samples:
            return {}
        
        gpu_utils = [s.gpu_util for s in samples]
        power_vals = [s.power_w for s in samples if s.power_w > 0]
        temps = [s.temp_c for s in samples if s.temp_c > 0]
        memory_vals = [s.memory_used_mb for s in samples]
        
        return {
            "sample_count": len(samples),
            "duration_s": round(samples[-1].timestamp - samples[0].timestamp, 3),
            "gpu_util_mean": round(statistics.mean(gpu_utils), 2) if gpu_utils else 0,
            "gpu_util_max": max(gpu_utils) if gpu_utils else 0,
            "gpu_util_min": min(gpu_utils) if gpu_utils else 0,
            "power_mean": round(statistics.mean(power_vals), 2) if power_vals else 0,
            "power_max": max(power_vals) if power_vals else 0,
            "temp_mean": round(statistics.mean(temps), 2) if temps else 0,
            "temp_max": max(temps) if temps else 0,
            "memory_peak_mb": max(memory_vals) if memory_vals else 0,
            "memory_avg_mb": round(statistics.mean(memory_vals), 2) if memory_vals else 0,
        }

    def _save_result(self, result: BenchmarkResult):
        """保存详细结果到JSON文件"""
        filename = f"{result.task_name}_{result.model_name}_{int(time.time())}.json"
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2, ensure_ascii=False)

    def _append_summary_csv(self, result: BenchmarkResult):
        """追加一行到汇总CSV"""
        csv_file = self.output_dir / "summary.csv"
        file_exists = csv_file.exists()
        
        fieldnames = [
            "timestamp", "task_name", "model_name", "precision", "status", 
            "duration_s", "gpu_util_mean", "gpu_util_max", "power_mean", 
            "power_max", "temp_mean", "temp_max", "memory_peak_mb"
        ]
        
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            
            row = {
                "timestamp": result.timestamp,
                "task_name": result.task_name,
                "model_name": result.model_name,
                "precision": result.precision,
                "status": result.status,
                "duration_s": result.duration_s,
                "gpu_util_mean": result.gpu_stats.get("gpu_util_mean", 0),
                "gpu_util_max": result.gpu_stats.get("gpu_util_max", 0),
                "power_mean": result.gpu_stats.get("power_mean", 0),
                "power_max": result.gpu_stats.get("power_max", 0),
                "temp_mean": result.gpu_stats.get("temp_mean", 0),
                "temp_max": result.gpu_stats.get("temp_max", 0),
                "memory_peak_mb": result.gpu_stats.get("memory_peak_mb", 0),
            }
            writer.writerow(row)
