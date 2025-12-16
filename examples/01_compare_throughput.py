# Copyright 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example 1: Compare Throughput
==============================

This example shows how to compare different implementations using a custom metric.

New concepts:
- Multiple `nsight.annotate()` blocks to profile different kernels
- Using `@nsight.annotate()` as a function decorator (alternative to context manager)
- Using the `metrics` parameter to collect a specific Nsight Compute metric (DRAM throughput instead of execution time)
- Using `print_data=True` to print the collected dataframe to the terminal
"""

import torch

import nsight


# You can use @nsight.annotate as a decorator on functions!
@nsight.annotate("torch.einsum")
def einsum_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matrix multiplication using einsum - annotated with decorator."""
    return torch.einsum("ij,jk->ik", a, b)


@nsight.analyze.plot(
    "01_compare_throughput.png",
    plot_type="bar",
)
@nsight.analyze.kernel(
    runs=10,
    # Collect DRAM throughput as percentage of peak instead of time
    metrics=["dram__throughput.avg.pct_of_peak_sustained_elapsed"],
)
def benchmark_matmul_throughput(n: int) -> None:
    """
    Compare DRAM throughput of different matrix multiplication methods.

    Note: All three methods call the same cuBLAS kernel, so throughput
    should be nearly identical. This example demonstrates how to use
    different metrics - try other operations to see more variation!
    """
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")

    # Method 1: Using @ operator with context manager
    with nsight.annotate("@-operator"):
        _ = a @ b

    # Method 2: Using torch.matmul with context manager
    with nsight.annotate("torch.matmul"):
        _ = torch.matmul(a, b)

    # Method 3: Using function with @nsight.annotate decorator
    _ = einsum_matmul(a, b)


def main() -> None:
    result = benchmark_matmul_throughput(2048)
    print(result.to_dataframe())
    print("✓ Benchmark complete! Check '01_compare_throughput.png'")
    print("\nTip: Run 'ncu --query-metrics' to see all available metrics!")


if __name__ == "__main__":
    main()
