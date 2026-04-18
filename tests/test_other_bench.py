from __future__ import annotations

import subprocess

from tensorforge.other_bench import ASRBenchmark, CVBenchmark, DiffusionBenchmark


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


def test_asr_worker_called_process_error_has_stage_and_details(monkeypatch):
    bench = ASRBenchmark(audio_files=["a.wav"], output_dir="results/test")

    def _raise_called_process_error(*_args, **_kwargs):
        raise subprocess.CalledProcessError(2, ["python"], output=b"traceback: boom")

    monkeypatch.setattr(subprocess, "check_output", _raise_called_process_error)
    result = bench.run_task()
    assert result["error_stage"] == "asr_worker_called_process"
    assert result["returncode"] == 2
    assert "boom" in result["details"]


def test_diffusion_worker_invalid_output_sets_parse_stage(monkeypatch):
    bench = DiffusionBenchmark(output_dir="results/test", local_model_path="/tmp/model")

    monkeypatch.setattr(subprocess, "check_output", lambda *_args, **_kwargs: b"not-json")
    result = bench.run_task()
    assert result["error_stage"] == "diffusion_worker_parse_output"
    assert "not-json" in result["details"]
