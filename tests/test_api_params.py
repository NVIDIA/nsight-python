# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
pytest tests for API parameter validation.
"""

import re
from typing import Any, Callable, Dict

import pytest
import torch

import nsight

# powers of two, 1k - 4k
sizes = [(2**i,) for i in range(10, 13)]

# Test scenario definitions
TEST_SCENARIOS: Dict[str, Dict[str, Any]] = {
    # ==========================================================================
    # Config validation error scenarios
    # ==========================================================================
    "no_configs": {
        "description": "No configs provided - should error",
        "decorator_configs": None,
        "runtime_configs": None,
        "expected": "You have provided no configs. Provide configs at decoration time or at runtime.",
    },
    "empty_configs": {
        "description": "Empty configs list - should error",
        "decorator_configs": [],
        "expected": "configs cannot be empty",
    },
    "both_configs": {
        "description": "Both decorator and runtime configs - should error",
        "runtime_configs": [(1024,), (2048,)],
        "expected": "You have provided configs at decoration time and at runtime. Provide configs at decoration time or at runtime.",
        # Note: decorator_configs is not set here, so the default 'sizes' will be used.
    },
    "mismatched_lengths": {
        "description": "Configs with different argument counts - should error",
        "decorator_configs": [(1024,), (2048, 512), (4096,)],
        "expected": "All configs must have the same number of arguments. Found lengths: [1, 2, 1]",
    },
    "wrong_arg_count": {
        "description": "Configs have wrong number of arguments for function - should error",
        "decorator_configs": [(1024, 512), (2048, 1024)],
        "expected": "Configs have 2 arguments, but function expects 1",
    },
    # ==========================================================================
    # RuntimeError scenarios
    # ==========================================================================
    "inconsistent_kernel_counts": {
        "description": "RuntimeError: Expect same number of kernels per run",
        "decorator_configs": [(1024,), (2048,)],
        "benchmark_type": "variable_kernels",
        "runs": 1,
        "expected": "Expect same number of kernels per run",
    },
    "multiple_kernels_no_combine": {
        "description": "RuntimeError: More than one kernel is launched within annotation",
        "decorator_configs": [(2048,)],
        "benchmark_type": "multiple_kernels",
        "runs": 1,
        "expected": "More than one .* kernel is launched",
    },
}


def create_benchmark(
    benchmark_type: str,
    kernel_kwargs: Dict[str, Any],
    params: Dict[str, Any] | None = None,
) -> Any:
    """
    Helper function to create benchmark based on type.
    Used by both pytest and CLI.
    """
    if benchmark_type == "variable_kernels":
        # Benchmark that launches different numbers of kernels per config
        @nsight.analyze.kernel(**kernel_kwargs)
        def run_benchmark(n: int) -> None:
            a = torch.randn(n, n, device="cuda")
            b = torch.randn(n, n, device="cuda")

            with nsight.annotate("variable_kernels"):
                if n >= 2048:
                    _ = a @ b  # Kernel 1: matmul
                    _ = a + b  # Kernel 2: elementwise add
                else:
                    _ = a @ b  # Kernel 1: matmul

    elif benchmark_type == "multiple_kernels":
        # Benchmark that launches multiple different kernels in one annotation
        @nsight.analyze.kernel(**kernel_kwargs)
        def run_benchmark(n: int) -> None:
            a = torch.randn(n, n, device="cuda")
            b = torch.randn(n, n, device="cuda")
            # Launch MULTIPLE DIFFERENT kernels in the same annotation
            with nsight.annotate("multiple_kernels"):
                _ = a @ b  # Kernel 1: matmul
                _ = a + b  # Kernel 2: elementwise add

    else:
        # Default benchmark with multiple annotations
        if params is None:
            raise ValueError("params cannot be None")

        @nsight.annotate(params["annotate2"])
        def einsum(a: torch.Tensor, b: torch.Tensor) -> Any:
            return torch.einsum("ij,jk->ik", a, b)

        @nsight.analyze.kernel(**kernel_kwargs)
        def run_benchmark(n: int) -> None:
            a = torch.randn(n, n, device="cuda")
            b = torch.randn(n, n, device="cuda")

            with nsight.annotate(params["annotate1"]):
                _ = a @ b

            einsum(a, b)

            with nsight.annotate(params["annotate3"]):
                _ = torch.nn.functional.linear(a, b)

    return run_benchmark


# =============================================================================
# pytest Tests
# =============================================================================


@pytest.mark.parametrize(
    "scenario_name",
    [
        "no_configs",
        "empty_configs",
        "both_configs",
        "mismatched_lengths",
        "wrong_arg_count",
    ],
)  # type: ignore[untyped-decorator]
def test_config_validation_errors(scenario_name: str) -> None:
    """Test that config validation errors are raised correctly."""
    scenario = TEST_SCENARIOS[scenario_name]

    # Build kernel kwargs
    kernel_kwargs = {
        "runs": scenario.get("runs", 1),
        "metrics": ["dram__bytes.sum.per_second"],
        "replay_mode": "kernel",
        "clock_control": "none",
        "cache_control": "all",
        "thermal_control": True,
        "output": "quiet",
    }

    if "decorator_configs" in scenario:
        if scenario["decorator_configs"] is not None:
            kernel_kwargs["configs"] = scenario["decorator_configs"]
    else:
        kernel_kwargs["configs"] = sizes

    # Create params for benchmark creation
    params = {
        "annotate1": "matmul",
        "annotate2": "einsum",
        "annotate3": "linear",
    }

    benchmark_type = scenario.get("benchmark_type", "default")
    run_benchmark = create_benchmark(benchmark_type, kernel_kwargs, params)

    # Assert error is raised
    with pytest.raises(Exception) as exc_info:
        if scenario.get("runtime_configs"):
            run_benchmark(configs=scenario["runtime_configs"])
        else:
            run_benchmark()

    # Verify error message matches expected
    error_msg = str(exc_info.value)
    expected_msg = scenario["expected"]

    assert re.search(
        re.escape(expected_msg), error_msg, re.IGNORECASE
    ), f"Expected error message to contain: '{expected_msg}'\nActual error: '{error_msg}'"


@pytest.mark.parametrize(
    "scenario_name",
    [
        "inconsistent_kernel_counts",
        "multiple_kernels_no_combine",
    ],
)  # type: ignore[untyped-decorator]
def test_runtime_extraction_errors(scenario_name: str) -> None:
    """Test that runtime extraction errors are raised correctly."""
    scenario = TEST_SCENARIOS[scenario_name]

    kernel_kwargs = {
        "runs": 1,
        "metrics": ["dram__bytes.sum.per_second"],
        "configs": scenario["decorator_configs"],
        "replay_mode": "kernel",
        "clock_control": "none",
        "cache_control": "all",
        "thermal_control": True,
        "output": "quiet",
    }

    benchmark_type = scenario.get("benchmark_type", "default")
    run_benchmark = create_benchmark(benchmark_type, kernel_kwargs)

    # Assert RuntimeError is raised with expected message
    with pytest.raises(RuntimeError) as exc_info:
        run_benchmark()

    # Verify error message matches expected
    error_msg = str(exc_info.value)
    expected_msg = scenario["expected"]

    assert re.search(
        expected_msg, error_msg, re.IGNORECASE
    ), f"Expected error message to match pattern: '{expected_msg}'\nActual error: '{error_msg}'"
