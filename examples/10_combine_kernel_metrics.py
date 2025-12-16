# Copyright 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example 10: Multiple Kernels per Run with Combined Metrics
===========================================================

This example shows how to profile multiple kernels in a single run and combine their metrics.

New concepts:
- Using `combine_kernel_metrics` to aggregate metrics from multiple kernels
- Summing metrics from consecutive kernel executions
"""

import torch

import nsight

# Define configuration sizes
sizes = [(2**i,) for i in range(10, 13)]


@nsight.analyze.plot(
    filename="10_combine_kernel_metrics.png",
    ylabel="Total Cycles (Sum of 3 Kernels)",
    annotate_points=True,
)
@nsight.analyze.kernel(
    configs=sizes,
    runs=7,
    combine_kernel_metrics=lambda x, y: x + y,  # Sum metrics from multiple kernels
    metrics=[
        "sm__cycles_elapsed.avg",
    ],
)
def benchmark_multiple_kernels(n: int) -> None:
    """
    Benchmark three matrix multiplications in a single run.

    Executes three matmul operations within one profiled context,
    demonstrating metric combination across kernels.

    Args:
        n: Matrix size (n x n)
    """
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")

    with nsight.annotate("test"):
        # Three consecutive kernel executions
        _ = a @ b  # Kernel 1
        _ = a @ b  # Kernel 2
        _ = a @ b  # Kernel 3


def main() -> None:
    result = benchmark_multiple_kernels()
    print(result.to_dataframe())
    print("\n✓ Total Cycles benchmark complete! Check '10_combine_kernel_metrics.png'")


if __name__ == "__main__":
    main()
