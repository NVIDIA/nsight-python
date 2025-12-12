# Copyright 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example 5: Subplots and Faceting
=================================

This example shows how to create subplot grids for complex comparisons.

New concepts:
- Using `row_panels` to create subplot rows
- Using `col_panels` to create subplot columns
- Organizing complex multi-parameter sweeps visually
"""

import itertools
from typing import Any

import torch

import nsight

# Sweep over size, dtype, and transpose
sizes = [2048, 4096, 8192]
dtypes = [torch.float32, torch.float16]
transpose = [False, True]

configs = list(itertools.product(sizes, dtypes, transpose))


def compute_tflops(time_ns: float, *conf: Any) -> float:
    """Compute TFLOPs/s using *conf to extract only what we need."""
    n: int = conf[0]  # Extract size (dtype and transpose not needed)
    flops = 2 * n * n * n
    tflops: float = flops / (time_ns / 1e9) / 1e12
    return tflops


@nsight.analyze.plot(
    filename="05_subplots.png",
    title="Matrix Multiplication Performance",
    ylabel="TFLOPs/s",
    row_panels=["dtype"],  # Create row for each dtype
    col_panels=["transpose"],  # Create column for each transpose setting
    annotate_points=True,
)
@nsight.analyze.kernel(configs=configs, runs=10, derive_metric=compute_tflops)
def benchmark_with_subplots(n: int, dtype: torch.dtype, transpose: bool) -> None:
    """
    Benchmark with subplots organized by dtype and transpose.
    """
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)

    # Optionally transpose the second matrix
    if transpose:
        b = b.T

    with nsight.annotate("matmul"):
        _ = a @ b


def main() -> None:
    benchmark_with_subplots()
    print("✓ Subplot benchmark complete! Check '05_subplots.png'")


if __name__ == "__main__":
    main()
