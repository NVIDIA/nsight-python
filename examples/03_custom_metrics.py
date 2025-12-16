# Copyright 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example 3: Custom Metrics (TFLOPs)
===================================

This example shows how to compute custom metrics from timing data.

New concepts:
- Using `derive_metric` to compute custom values (e.g., TFLOPs)
- Customizing plot labels with `ylabel`
- The `annotate_points` parameter to show values on the plot
"""

import torch

import nsight

sizes = [(2**i,) for i in range(11, 14)]


def compute_tflops(time_ns: float, n: int) -> float:
    """
    Compute TFLOPs/s for matrix multiplication.

    Custom metric function signature:
    - First argument: the measured metric (time in nanoseconds by default)
    - Remaining arguments: must match the decorated function's signature

    In this example:
    - time_ns: The measured metric (gpu__time_duration.sum in nanoseconds)
    - n: Matches the 'n' parameter from benchmark_tflops(n)

    If your function was benchmark(size, dtype, batch), your metric function
    would be: my_metric(time_ns, size, dtype, batch)

    Args:
        time_ns: Kernel execution time in nanoseconds (automatically passed)
        n: Matrix size (n x n) - matches benchmark_tflops parameter

    Returns:
        TFLOPs/s (higher is better)
    """
    # Matrix multiplication: 2*n^3 FLOPs (n^3 multiplies + n^3 adds)
    flops = 2 * n * n * n
    # Convert ns to seconds and FLOPs to TFLOPs
    tflops = flops / (time_ns / 1e9) / 1e12
    return tflops


@nsight.analyze.plot(
    filename="03_custom_metrics.png",
    ylabel="Performance (TFLOPs/s)",  # Custom y-axis label
    annotate_points=True,  # Show values on the plot
)
@nsight.analyze.kernel(
    configs=sizes, runs=10, derive_metric=compute_tflops  # Use custom metric
)
def benchmark_tflops(n: int) -> None:
    """
    Benchmark matmul and display results in TFLOPs/s.
    """
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")

    with nsight.annotate("matmul"):
        _ = a @ b


def main() -> None:
    result = benchmark_tflops()
    print(result.to_dataframe())
    print("✓ TFLOPs benchmark complete! Check '03_custom_metrics.png'")


if __name__ == "__main__":
    main()
