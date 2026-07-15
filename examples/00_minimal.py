# Copyright 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example 0: Minimal Example
===========================

This is the absolute minimal example to get started with Nsight Python.
It shows the core concepts:
- Using `@nsight.analyze.kernel` to profile a function
- Using `with nsight.annotate()` to mark a kernel of interest
- Accessing the `ProfileResults` returned by the decorated function
"""

import torch

import nsight


@nsight.analyze.kernel
def benchmark_matmul(n: int) -> None:
    """
    The simplest possible benchmark.
    We create two matrices and multiply them.
    """
    # Create two NxN matrices on GPU
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")

    # Mark the operation we want to profile
    with nsight.annotate("matmul"):
        _ = a @ b


def main() -> None:
    # Decorated benchmark functions return ProfileResults.
    profile_results = benchmark_matmul(1024)
    print(
        profile_results.to_dataframe()[
            ["Annotation", "n", "Metric", "Unit", "AvgValue", "NumRuns", "GPU"]
        ]
    )
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
