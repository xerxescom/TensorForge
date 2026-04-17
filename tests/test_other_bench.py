from __future__ import annotations

from tensorforge.other_bench import CVBenchmark


def test_cv_benchmark_batch_size_is_clamped():
    bench = CVBenchmark(batch_size=0, output_dir="results/test")
    assert bench.batch_size == 1


def test_cv_benchmark_worker_script_uses_batched_inputs():
    bench = CVBenchmark(n_frames=10, batch_size=4, output_dir="results/test")
    script = bench._build_script()
    assert "for start in range(0, total_frames, batch_size):" in script
    assert "inputs = [frame] * current_batch" in script
    assert '"batch_per_s"' in script
