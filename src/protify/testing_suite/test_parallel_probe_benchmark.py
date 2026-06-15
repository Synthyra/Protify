from types import SimpleNamespace

import pytest
import torch

try:
    from src.protify.scripts import benchmark_parallel_probes as benchmark
except ImportError:
    try:
        from protify.scripts import benchmark_parallel_probes as benchmark
    except ImportError:
        from ..scripts import benchmark_parallel_probes as benchmark


def _args(**overrides):
    values = {
        "num_samples": 12,
        "input_size": 4,
        "hidden_size": 8,
        "num_labels": 3,
        "num_runs": 3,
        "batch_size": 5,
        "epochs": 1,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "n_layers": 0,
        "dropout": 0.0,
        "seed": 17,
        "task_type": "singlelabel",
        "device": "cpu",
        "data_on_device": False,
        "parallel_batch_mode": "shared",
        "parallel_index_strategy": "permutation",
        "parallel_max_group_size": None,
        "training_flop_multiplier": 3,
        "plan_only": False,
    }
    for key, value in overrides.items():
        values[key] = value
    return SimpleNamespace(**values)


def test_benchmark_validate_args_accepts_nominal_config() -> None:
    args = _args()

    assert benchmark.validate_args(args) is args


def test_benchmark_validate_args_rejects_invalid_configs() -> None:
    with pytest.raises(AssertionError, match="num_runs"):
        benchmark.validate_args(_args(num_runs=1))

    with pytest.raises(AssertionError, match="dropout"):
        benchmark.validate_args(_args(dropout=1.0))

    with pytest.raises(AssertionError, match="num_labels"):
        benchmark.validate_args(_args(num_labels=1))

    with pytest.raises(AssertionError, match="parallel_max_group_size"):
        benchmark.validate_args(_args(parallel_max_group_size=0))

    with pytest.raises(AssertionError, match="training_flop_multiplier"):
        benchmark.validate_args(_args(training_flop_multiplier=0))


def test_benchmark_builds_synthetic_plan_summary() -> None:
    args = _args(
        num_runs=4,
        parallel_batch_mode="run_specific",
        parallel_index_strategy="affine",
        training_flop_multiplier=4,
    )
    plan = benchmark.build_synthetic_plan(args)
    summary = benchmark.plan_summary_dict(plan, args)

    assert summary["total_runs"] == 4
    assert summary["trainer_invocations"] == 1
    assert summary["invocation_reduction"] == 3
    assert summary["compression_ratio"] == pytest.approx(4.0)
    assert summary["vectorized_runs"] == 4
    assert summary["sequential_runs"] == 0
    assert summary["total_parameter_count"] > 0
    assert summary["total_training_state_bytes"] > 0
    assert summary["peak_group_training_state_bytes"] == summary["total_training_state_bytes"]
    assert summary["total_batch_activation_bytes"] > 0
    assert summary["peak_group_batch_activation_bytes"] == summary["total_batch_activation_bytes"]
    assert summary["total_run_specific_index_bytes"] == 0
    assert summary["peak_group_estimated_peak_bytes"] > summary["peak_group_training_state_bytes"]
    assert summary["total_forward_flops_per_batch"] > 0
    assert summary["peak_group_forward_flops_per_batch"] == summary["total_forward_flops_per_batch"]
    assert summary["total_training_flops_per_batch"] == summary["total_forward_flops_per_batch"] * 4
    assert summary["peak_group_training_flops_per_batch"] == summary["peak_group_forward_flops_per_batch"] * 4
    assert summary["unknown_estimate_group_count"] == 0


def test_benchmark_plan_only_result_reports_static_plan_without_timings() -> None:
    args = _args(num_runs=4, plan_only=True, device="cuda")
    plan = benchmark.build_synthetic_plan(args)

    result = benchmark.plan_only_result(args, plan)

    assert result["plan_only"] is True
    assert result["requested_device"] == "cuda"
    assert result["num_runs"] == 4
    assert result["plan"]["total_runs"] == 4
    assert result["plan"]["trainer_invocations"] == 1
    assert result["comparison"]["sequential_trainer_invocations"] == 4
    assert result["comparison"]["parallel_trainer_invocations"] == 1
    assert result["comparison"]["trainer_invocation_speedup_ceiling"] == pytest.approx(4.0)
    assert result["comparison"]["speedup_formulas"] == {
        "sequential_total_seconds": "sequential_seconds",
        "parallel_total_seconds": "parallel_seconds",
        "sequential_seconds_per_run": "sequential_seconds / num_runs",
        "parallel_seconds_per_run": "parallel_seconds / num_runs",
        "wall_clock_speedup": "sequential_seconds / parallel_seconds",
        "per_run_speedup": "sequential_seconds_per_run / parallel_seconds_per_run",
        "trainer_invocation_speedup_ceiling": "sequential_trainer_invocations / parallel_trainer_invocations",
    }
    assert result["comparison"]["runtime_metric_keys"] == [
        "sequential_seconds",
        "parallel_seconds",
        "sequential_seconds_per_run",
        "parallel_seconds_per_run",
        "speedup",
        "per_run_speedup",
        "sequential_peak_memory_bytes",
        "parallel_peak_memory_bytes",
        "parallel_index_memory_bytes",
    ]
    assert result["comparison"]["hardware_metric_keys"] == [
        "gpu_utilization_percent",
        "sm_occupancy_percent",
        "memory_bandwidth_percent",
        "peak_gpu_memory_bytes",
        "cpu_utilization_percent",
        "dataloader_wait_seconds",
    ]
    assert "device" not in result
    assert "sequential_seconds" not in result
    assert "parallel_seconds" not in result
    assert "speedup" not in result


def test_benchmark_builds_chunked_synthetic_plan_summary() -> None:
    args = _args(num_runs=5, parallel_max_group_size=2)
    plan = benchmark.build_synthetic_plan(args)
    summary = benchmark.plan_summary_dict(plan, args)

    assert summary["total_runs"] == 5
    assert summary["trainer_invocations"] == 3
    assert summary["invocation_reduction"] == 2
    assert summary["compression_ratio"] == pytest.approx(5.0 / 3.0)
    assert summary["vectorized_runs"] == 4
    assert summary["sequential_runs"] == 1
    assert summary["total_parameter_count"] > 0
    assert summary["peak_group_training_state_bytes"] < summary["total_training_state_bytes"]
    assert summary["peak_group_batch_activation_bytes"] < summary["total_batch_activation_bytes"]
    assert summary["peak_group_estimated_peak_bytes"] > summary["peak_group_training_state_bytes"]
    assert summary["peak_group_forward_flops_per_batch"] < summary["total_forward_flops_per_batch"]
    assert summary["peak_group_training_flops_per_batch"] < summary["total_training_flops_per_batch"]
    assert summary["unknown_estimate_group_count"] == 0
    assert benchmark.parallel_seed_groups(args) == ((17, 18), (19, 20), (21,))


def test_benchmark_plan_summary_counts_permutation_index_memory() -> None:
    args = _args(num_runs=4, parallel_batch_mode="run_specific", parallel_index_strategy="permutation")
    plan = benchmark.build_synthetic_plan(args)

    summary = benchmark.plan_summary_dict(plan, args)

    assert summary["total_run_specific_index_bytes"] == args.num_runs * args.num_samples * 8
    assert summary["peak_group_estimated_peak_bytes"] > summary["peak_group_training_state_bytes"]


def test_benchmark_shared_parallel_loader_keeps_base_batch_shape() -> None:
    args = _args(parallel_batch_mode="shared")
    dataset = benchmark.make_dataset(args, torch.device("cpu"))
    loader, index_memory_bytes = benchmark.make_parallel_loader(args, dataset, benchmark.run_seeds(args))
    embeddings, labels = next(iter(loader))

    assert index_memory_bytes == 0
    assert embeddings.shape == (args.batch_size, args.input_size)
    assert labels.shape == (args.batch_size,)


def test_benchmark_run_specific_parallel_loader_uses_run_dimension() -> None:
    args = _args(parallel_batch_mode="run_specific", parallel_index_strategy="permutation")
    dataset = benchmark.make_dataset(args, torch.device("cpu"))
    seeds = benchmark.run_seeds(args)
    loader, index_memory_bytes = benchmark.make_parallel_loader(args, dataset, seeds)
    embeddings, labels = next(iter(loader))

    assert index_memory_bytes == args.num_runs * args.num_samples * 8
    assert embeddings.shape == (args.batch_size, args.num_runs, args.input_size)
    assert labels.shape == (args.batch_size, args.num_runs)


def test_benchmark_run_specific_affine_loader_has_zero_index_memory() -> None:
    args = _args(parallel_batch_mode="run_specific", parallel_index_strategy="affine")
    dataset = benchmark.make_dataset(args, torch.device("cpu"))
    seeds = benchmark.run_seeds(args)
    loader, index_memory_bytes = benchmark.make_parallel_loader(args, dataset, seeds)
    embeddings, labels = next(iter(loader))

    assert index_memory_bytes == 0
    assert embeddings.shape == (args.batch_size, args.num_runs, args.input_size)
    assert labels.shape == (args.batch_size, args.num_runs)
