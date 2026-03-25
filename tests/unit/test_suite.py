from src.benchmarks.suite import ConcurrentStressTest


def test_calculate_throughput_reads_nested_metrics() -> None:
    stress = ConcurrentStressTest(tasks=[])

    metric = {
        "task_type": "llm",
        "status": "success",
        "metrics": {
            "status": "success",
            "tokens_per_s": 25.0,
        },
    }

    assert stress._calculate_throughput(metric) == 25.0


def test_analyze_throughput_avoids_division_by_zero() -> None:
    stress = ConcurrentStressTest(tasks=[])

    concurrent_metrics = [
        {
            "task_type": "llm",
            "status": "success",
            "metrics": {"status": "success", "tokens_per_s": 0.0},
        }
    ]
    baselines = {
        "llm": {
            "elapsed_s": 1.0,
            "metrics": {"status": "success", "tokens_per_s": 0.0},
        }
    }

    analysis = stress._analyze_throughput(concurrent_metrics, baselines)

    assert analysis["throughput_degradation"]["llm"] == 0.0
    assert analysis["efficiency_ratio"]["llm"] == 0.0
