from __future__ import annotations

from tensorforge.other_bench import CVBenchmark


def test_cv_worker_script_reports_effective_batch_size(tmp_path):
    bench = CVBenchmark(
        model_name="yolov8n",
        precision="fp16",
        n_frames=10,
        image_size=320,
        batch_size=3,
        output_dir=str(tmp_path),
        warmup_s=0,
    )
    cmd = bench._build_worker_command()
    assert "tensorforge.cv_bench_worker" in cmd
    assert "--n-frames" in cmd
    assert cmd[cmd.index("--n-frames") + 1] == "10"
