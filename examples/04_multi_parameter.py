# Copyright 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example 4: Multi-Parameter Sweeps
==================================

This example shows how to sweep over multiple parameters simultaneously.

New concepts:
- Creating multi-parameter configurations with itertools.product
- Functions accepting multiple parameters
- Using `*conf` in custom metric functions to flexibly handle config parameters
- Extracting only the config parameters you need
"""

import itertools
from typing import Any

import torch

import nsight

# Sweep over both matrix sizes and data types
sizes = [2048, 4096, 8192]
dtypes = [torch.float32, torch.float16]

# Create all combinations: [(512, float32), (512, float16), (1024, float32), ...]
configs = list(itertools.product(sizes, dtypes))


def compute_tflops(time_ns: float, *conf: Any) -> float:
    """
    Compute TFLOPs/s.

    Alternative signature using *conf:
    - time_ns: The measured metric (automatically passed as first argument)
    - *conf: Captures all config parameters as a tuple

    This is more flexible than explicitly listing all parameters.
    We can then extract only what we need:
    """
    n: int = conf[0]  # First config parameter (size)
    # dtype = conf[1]  # Second config parameter (not needed for this calculation)

    flops = 2 * n * n * n
    tflops: float = flops / (time_ns / 1e9) / 1e12
    return tflops


@nsight.analyze.plot(
    filename="04_multi_parameter.png",
    ylabel="Performance (TFLOPs/s)",
    annotate_points=True,
)
@nsight.analyze.kernel(configs=configs, runs=10, derive_metric=compute_tflops)
def benchmark_multi_param(
    n: int, dtype: torch.dtype
) -> None:  # Function now takes multiple parameters
    """
    Benchmark across different sizes and data types.
    """
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)

    with nsight.annotate("matmul"):
        _ = a @ b


def main() -> None:
    benchmark_multi_param()
    print("✓ Multi-parameter sweep complete! Check '04_multi_parameter.png'")


if __name__ == "__main__":
    main()
