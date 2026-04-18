from __future__ import annotations

from tensorforge.other_bench import ASRBenchmark, CVBenchmark


def test_cv_benchmark_batch_size_is_clamped():
    bench = CVBenchmark(batch_size=0, output_dir="results/test")
    assert bench.batch_size == 1


def test_cv_benchmark_worker_script_uses_batched_inputs():
    bench = CVBenchmark(n_frames=10, batch_size=4, output_dir="results/test")
    cmd = bench._build_worker_command()
    assert "tensorforge.cv_bench_worker" in cmd
    assert "--batch-size" in cmd
    assert cmd[cmd.index("--batch-size") + 1] == "4"


def test_asr_benchmark_worker_script_includes_levenshtein_wer_and_compat_field():
    bench = ASRBenchmark(
        audio_files=["a.wav"],
        ground_truths=["hello world"],
        output_dir="results/test",
    )
    cmd = bench._build_worker_command(synthetic=False)
    assert "tensorforge.asr_bench_worker" in cmd
    assert "--audio-files-json" in cmd
    assert "--ground-truths-json" in cmd
