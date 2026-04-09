from __future__ import annotations

import time

from tensorforge.collector import GPUSample, GPUSampler


def test_smart_sampling_keeps_first_last_and_limits():
    sampler = GPUSampler(interval_s=0.5, gpu_index=0, adaptive_sampling=True)
    t0 = time.time()
    samples = [
        GPUSample(
            timestamp=t0 + i,
            gpu_util=float(i % 100),
            mem_used_mb=float(i),
            mem_total_mb=1000.0,
            power_w=float(i % 200),
            temp_c=float(40 + (i % 20)),
            sm_clock_mhz=1000.0,
            mem_clock_mhz=5000.0,
        )
        for i in range(200)
    ]
    out = sampler.smart_sampling(samples, max_samples=50)
    assert len(out) <= 50
    assert out[0].timestamp == samples[0].timestamp
    assert out[-1].timestamp == samples[-1].timestamp


def test_summarize_returns_expected_keys():
    t0 = time.time()
    samples = [
        GPUSample(
            timestamp=t0,
            gpu_util=10.0,
            mem_used_mb=100.0,
            mem_total_mb=1000.0,
            power_w=50.0,
            temp_c=60.0,
            sm_clock_mhz=1000.0,
            mem_clock_mhz=5000.0,
        ),
        GPUSample(
            timestamp=t0 + 1,
            gpu_util=90.0,
            mem_used_mb=200.0,
            mem_total_mb=1000.0,
            power_w=150.0,
            temp_c=80.0,
            sm_clock_mhz=1100.0,
            mem_clock_mhz=5100.0,
        ),
    ]
    stats = GPUSampler.summarize(samples)
    assert stats["sample_count"] == 2
    assert "gpu_util_%" in stats
    assert stats["gpu_util_%"]["max"] == 90.0
