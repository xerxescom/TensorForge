"""
GPU Benchmark - Automated Data Collector.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .tf_logger import logger

IS_WINDOWS = sys.platform == "win32"

if not IS_WINDOWS and sys.platform != "linux":
    raise SystemError(
        f"Unsupported platform: {sys.platform}. Only Windows and Linux are supported."
    )


def _find_nvidia_smi() -> str:
    if IS_WINDOWS:
        try:
            subprocess.check_output(
                ["nvidia-smi", "--version"],
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            return "nvidia-smi"
        except Exception:
            pass

        candidates = [
            r"C:\Windows\System32\nvidia-smi.exe",
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        ]
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
    return "nvidia-smi"


def _subprocess_kwargs() -> dict:
    if IS_WINDOWS:
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


try:
    NVIDIA_SMI: str | None = _find_nvidia_smi()
except FileNotFoundError as _e:
    NVIDIA_SMI = None
    logger.warning(str(_e))


@dataclass
class GPUSample:
    timestamp: float
    gpu_util: float
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
    precision: str
    start_time: str
    end_time: str
    duration_s: float
    status: str = "success"
    error: str | None = None
    environment: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    gpu_stats: dict = field(default_factory=dict)
    raw_samples: list = field(default_factory=list)


class GPUSampler:
    def __init__(self, interval_s: float = 0.5, gpu_index: int = 0, adaptive_sampling: bool = True):
        logger.debug(
            f"[collector] Initializing GPUSampler: interval={interval_s}s, gpu={gpu_index}, adaptive={adaptive_sampling}"
        )
        self.interval_s = interval_s
        self.base_interval_s = interval_s
        self.gpu_index = gpu_index
        self.adaptive_sampling = adaptive_sampling
        self._samples: list[GPUSample] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._query_count = 0
        self._error_count = 0
        self._timeout_count = 0
        self._query_latency_ms: list[float] = []
        self._last_error: str | None = None
        self._last_gpu_util = 0.0

    def _query(self) -> tuple[GPUSample | None, str | None, float]:
        if NVIDIA_SMI is None:
            logger.error("[collector] nvidia-smi not available for GPU monitoring")
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
            sample = GPUSample(
                timestamp=time.time(),
                gpu_util=safe(vals[0]),
                mem_used_mb=safe(vals[1]),
                mem_total_mb=safe(vals[2]),
                power_w=safe(vals[3]),
                temp_c=safe(vals[4]),
                sm_clock_mhz=safe(vals[5]),
                mem_clock_mhz=safe(vals[6]),
            )
            return sample, None, latency_ms
        except subprocess.TimeoutExpired:
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.warning(f"[collector] nvidia-smi query timeout after {latency_ms:.1f}ms")
            return None, "nvidia-smi query timeout", latency_ms
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.error(f"[collector] nvidia-smi query failed: {e}")
            return None, str(e), latency_ms

    def start(self):
        logger.info(f"[collector] Starting GPU sampler for GPU {self.gpu_index}")
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
        if len(samples) <= max_samples:
            return samples

        result = [samples[0], samples[-1]]
        gpu_utils = [s.gpu_util for s in samples]
        power_vals = [s.power_w for s in samples]
        temp_vals = [s.temp_c for s in samples]

        peak_indices: set[int] = set()
        for values in [gpu_utils, power_vals, temp_vals]:
            peak_indices.add(values.index(max(values)))
            peak_indices.add(values.index(min(values)))

        remaining_slots = max_samples - len(result) - len(peak_indices)
        if remaining_slots > 0:
            step = max(len(samples) // remaining_slots, 1)
            uniform_indices = set(range(0, len(samples), step))
        else:
            uniform_indices = set()

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
        if not self.adaptive_sampling:
            return self.base_interval_s
        if current_util > 80.0:
            return self.base_interval_s * 0.4
        if current_util < 20.0:
            return self.base_interval_s * 2.0
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
                next_interval = self._adaptive_interval(sample.gpu_util)
            else:
                self._error_count += 1
                if error == "nvidia-smi query timeout":
                    self._timeout_count += 1
                self._last_error = error
                next_interval = self.base_interval_s
            self._stop_event.wait(next_interval)

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
            "gpu_util_%": _stats([s.gpu_util for s in samples]),
            "power_w": _stats([s.power_w for s in samples]),
            "temp_c": _stats([s.temp_c for s in samples]),
            "mem_used_mb": _stats([s.mem_used_mb for s in samples]),
            "sm_clock_mhz": _stats([s.sm_clock_mhz for s in samples]),
            "sample_count": len(samples),
        }

    def health(self) -> dict:
        avg_latency = (
            round(sum(self._query_latency_ms) / len(self._query_latency_ms), 3)
            if self._query_latency_ms
            else 0.0
        )
        drop_rate = round(self._error_count / self._query_count, 4) if self._query_count else 0.0
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
    mean_power = gpu_stats.get("power_w", {}).get("mean", 0)
    if mean_power > 0 and duration_s > 0:
        energy_j = mean_power * duration_s
        metrics["energy_j"] = round(energy_j, 2)
        if "tokens_generated" in metrics:
            metrics["tokens_per_joule"] = round(metrics["tokens_generated"] / energy_j, 4)
        if "total_steps" in metrics:
            metrics["steps_per_joule"] = round(metrics["total_steps"] / energy_j, 4)
    return metrics


class BenchmarkRunner:
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
        adaptive_sampling: bool = True,
        max_raw_samples: int = 1000,
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
        self.adaptive_sampling = adaptive_sampling
        self.max_raw_samples = max_raw_samples
        self._sampler = GPUSampler(
            sample_interval_s, gpu_index, adaptive_sampling=adaptive_sampling
        )

    def run_task(self) -> dict:
        raise NotImplementedError

    def run(self) -> BenchmarkResult:
        logger.info(f"[{self.task_name}] Warm-up {self.warmup_s}s ...")
        time.sleep(self.warmup_s)

        logger.info(f"[{self.task_name}] Starting GPU sampler ...")
        self._sampler.start()
        start_ts = time.time()
        start_str = datetime.now().isoformat(timespec="seconds")
        metrics: dict = {}
        status = "success"
        error: str | None = None

        logger.info(f"[{self.task_name}] Running task ...")
        try:
            metrics = self.run_task()
        except Exception as e:
            status = "failed"
            error = str(e)
            logger.exception(f"[{self.task_name}] [error] {error}")

        end_ts = time.time()
        end_str = datetime.now().isoformat(timespec="seconds")
        samples = self._sampler.stop(max_raw_samples=self.max_raw_samples)

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
            raw_samples=[asdict(s) for s in samples] if self.keep_raw_samples else [],
        )

        self._save(result)
        self._print_summary(result)
        return result

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
                out = (
                    subprocess.check_output(
                        [
                            NVIDIA_SMI,
                            f"--id={self.gpu_index}",
                            "--query-gpu=name,driver_version,memory.total,power.limit",
                            "--format=csv,noheader,nounits",
                        ],
                        stderr=subprocess.DEVNULL,
                        timeout=3,
                        **_subprocess_kwargs(),
                    )
                    .decode()
                    .strip()
                )
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
            "adaptive_sampling": self.adaptive_sampling,
            "max_raw_samples": self.max_raw_samples,
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

        json_path = self.output_dir / f"{stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2, ensure_ascii=False)

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
