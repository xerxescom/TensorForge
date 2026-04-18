import json

from tensorforge.run_suite import ConcurrentStressTest


def test_timed_diffusion_invokes_fixed_worker_with_cli_args(monkeypatch):
    bench = ConcurrentStressTest(tasks=[{"type": "diffusion", "model": "sdxl-turbo"}])

    captured: dict = {}

    def _fake_check_output(cmd, timeout, **kwargs):
        captured["cmd"] = cmd
        captured["timeout"] = timeout
        captured["kwargs"] = kwargs
        payload = {
            "measurement_mode": "cold_start",
            "model_load_s": 1.0,
            "warmup_s": 0.0,
            "inference_only_s": 2.0,
            "end_to_end_s": 3.0,
            "n_images": 1,
            "total_steps": 10,
            "it_per_s": 3.33,
            "it_per_s_end_to_end": 3.33,
            "it_per_s_inference_only": 5.0,
        }
        return json.dumps(payload).encode()

    monkeypatch.setattr("tensorforge.run_suite.subprocess.check_output", _fake_check_output)

    metrics = bench._timed_diffusion(model="sdxl-turbo", duration_s=12)

    assert metrics["total_steps"] == 10
    assert captured["cmd"][1:3] == ["-m", "tensorforge.diffusion_worker"]
    assert "-c" not in captured["cmd"]
    assert captured["cmd"][3:] == [
        "--model",
        "sdxl-turbo",
        "--duration",
        "12",
        "--mode",
        "cold_start",
    ]
    assert captured["timeout"] == 42
