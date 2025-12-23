# Copyright 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example 3: Custom Metrics (TFLOPs and Arithmetic Intensity)
============================================================

This example demonstrates two patterns for using `derive_metric` to compute
custom performance metrics from profiling data.

New concepts:
- Using `derive_metric` to compute custom values (e.g., TFLOPs)
- Customizing plot labels with `ylabel`
- The `annotate_points` parameter to show values on the plot

Additional insights on `derive_metric` usage patterns:
1. `derive_metric` can return either:
   - A single scalar value (e.g., TFLOPS)
   - A dictionary containing one or more derived metrics (e.g., {"TFLOPS": value} or
     {"TFLOPS": value, "ArithIntensity": value})
2. When using `derive_metric`, plotting requires explicit metric specification:
   - Without `derive_metric`: Only one collected metric exists -> plot(metric=None) works
   - With `derive_metric`: Multiple metrics exist -> MUST specify which metric to plot
3. How to specify which metric to plot in different scenarios:
   - For scalar returns: plot(metric="function_name")
   - For dictionary returns: plot(metric="dictionary_key")
"""

import torch

import nsight

# Matrix sizes to benchmark: 2^11, 2^12, 2^13
sizes = [(2**i,) for i in range(11, 14)]


# ------------------------------------------------------------------------------
# Pattern 1: Returning a single scalar value
# ------------------------------------------------------------------------------


def compute_tflops(time_ns: float, n: int) -> float:
    """
    Compute TFLOPS for matrix multiplication.

    This function demonstrates the first pattern: returning a single scalar value.

    Function signature convention for `derive_metric`:
    - First argument: the measured base metric (default: gpu__time_duration.sum in nanoseconds)
    - Remaining arguments: must match the decorated function's parameters

    Note: When `derive_metric` returns a single value, the plot decorator's
    `metric` parameter must be set to the FUNCTION NAME (as a string,
    "compute_tflops" in this case).

    Args:
        time_ns: Kernel execution time in nanoseconds (automatically passed)
        n: Matrix size (n x n) - matches benchmark_tflops parameter

    Returns:
        TFLOPS (higher is better)
    """
    # Matrix multiplication FLOPs: 2 * n^3 (n^3 multiplications + n^3 additions)
    flops = 2 * n * n * n

    # Compute TFLOPS
    tflops = flops / (time_ns / 1e9) / 1e12

    # This function can also return a directory of one metric, such as
    # {"TFLOPS": tflops}, but the "metric" of the plot decorator must be
    # set to "TFLOPS" instead of "compute_tflops".
    return tflops


@nsight.analyze.plot(
    filename="03_custom_metrics_tflops.png",
    metric="compute_tflops",  # Must match the function name of `derive_metric` when returning scalar
    ylabel="Performance (TFLOPS)",
    annotate_points=True,
)
@nsight.analyze.kernel(
    configs=sizes, runs=10, derive_metric=compute_tflops  # Single scalar return
)
def benchmark_tflops(n: int) -> None:
    """
    Benchmark matrix multiplication and display results in TFLOPS.

    This example shows:
    - When `derive_metric` returns a single value, the plot metric parameter
      must be the function name ("compute_tflops")
    - Without `derive_metric`, there's only one collected metric (time duration),
      we don't need to specify a metric because plot(metric=None) works by default
    - With `derive_metric`, we have >1 metrics (time duration + derived), so we must
      explicitly specify which metric to plot
    """
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")

    with nsight.annotate("matmul"):
        _ = a @ b


# ------------------------------------------------------------------------------
# Pattern 2: Returning a dictionary of metrics
# ------------------------------------------------------------------------------


def compute_tflops_and_arithmetic_intensity(time_ns: float, n: int) -> dict[str, float]:
    """
    Compute both TFLOPS and Arithmetic Intensity for matrix multiplication.

    This function demonstrates the second pattern: returning a dictionary
    containing multiple derived metrics.

    Important: When derive_metric returns a dictionary, the plot decorator's
    `metric` parameter must be set to a KEY from the dictionary (as a string).

    Note: A single scalar value could also be returned as a dictionary with
    one key-value pair for consistency, but returning the scalar directly is
    more concise.

    Args:
        time_ns: Kernel execution time in nanoseconds (automatically passed)
        n: Matrix size (n x n) - matches benchmark_tflops parameter

    Returns:
        Dictionary with two metrics (TFLOPS and ArithIntensity)
    """
    # Matrix multiplication FLOPs: 2 * n^3
    flops = 2 * n * n * n

    # Compute TFLOPS
    tflops = flops / (time_ns / 1e9) / 1e12

    # Memory access calculation:
    # - Input matrices: n * n each, Output matrix: n * n
    # - Float32 datatype (4 bytes per element)
    memory_bytes = (n * n + n * n + n * n) * 4

    # Arithmetic Intensity = FLOPs / Bytes accessed
    arithmetic_intensity = flops / memory_bytes

    return {
        "TFLOPS": tflops,
        "ArithIntensity": arithmetic_intensity,
    }


@nsight.analyze.plot(
    filename="03_custom_metrics_arith_intensity.png",
    metric="ArithIntensity",  # Must be a key from the returned dictionary
    ylabel="Arithmetic Intensity (FLOPs/Byte)",
    annotate_points=True,
)
@nsight.analyze.kernel(
    configs=sizes,
    runs=10,
    derive_metric=compute_tflops_and_arithmetic_intensity,  # Dictionary return
)
def benchmark_tflops_and_arithmetic_intensity(n: int) -> None:
    """
    Benchmark matrix multiplication with multiple derived metrics.

    This example shows:
    - When `derive_metric` returns a dictionary, the plot metric parameter
      must be a key from that dictionary (e.g., "ArithIntensity")
    - You can have multiple derived metrics but plot only one at a time
    - All derived metrics are available in the ProfileResults object
    """
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")

    with nsight.annotate("matmul"):
        _ = a @ b


def main() -> None:
    # Run single-metric benchmark
    print("Running TFLOPs benchmark (scalar return pattern)...")
    benchmark_tflops()
    print("✓ TFLOPs benchmark complete! Check '03_custom_metrics_tflops.png'\n")

    # Run multi-metric benchmark
    print("Running combined benchmark (dictionary return pattern)...")
    result = benchmark_tflops_and_arithmetic_intensity()
    print(result.to_dataframe())

    print("\n✓ TFLOPs and Arithmetic Intensity benchmark complete! ", end="")
    print("Check '03_custom_metrics_arith_intensity.png'")


if __name__ == "__main__":
    main()
