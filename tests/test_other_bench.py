from __future__ import annotations

from tensorforge.other_bench import ASRBenchmark, CVBenchmark


def test_cv_benchmark_batch_size_is_clamped():
    bench = CVBenchmark(batch_size=0, output_dir="results/test")
    assert bench.batch_size == 1


def test_cv_benchmark_worker_script_uses_batched_inputs():
    bench = CVBenchmark(n_frames=10, batch_size=4, output_dir="results/test")
    script = bench._build_script()
    assert "for start in range(0, total_frames, batch_size):" in script
    assert "inputs = [frame] * current_batch" in script
    assert '"batch_per_s"' in script


def test_asr_benchmark_worker_script_includes_levenshtein_wer_and_compat_field():
    bench = ASRBenchmark(
        audio_files=["a.wav"],
        ground_truths=["hello world"],
        output_dir="results/test",
    )
    script = bench._build_script()
    assert "def levenshtein_wer(ref_words, hyp_words):" in script
    assert '"wer_approx": wer_approx' in script
    assert '"wer": wer' in script
    assert '"wer_s": wer_s' in script
