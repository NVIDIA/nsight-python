# Copyright 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example 2: Parameter Sweep
===========================

This example shows how to sweep over different problem sizes.

New concepts:
- The `configs` parameter to sweep over different configurations
- How the benchmark function receives config parameters
- Automatic x-axis labeling based on parameter names
"""

import torch

import nsight

# Define problem sizes to test (powers of 2 from 512 to 4096)
sizes = [(2**i,) for i in range(11, 14)]  # [(2048,), (4096,), (8192,)]


@nsight.analyze.plot("02_parameter_sweep.png")
@nsight.analyze.kernel(configs=sizes, runs=10)
def benchmark_matmul_sizes(n: int) -> None:
    """
    Benchmark matrix multiplication across different sizes.
    The 'n' parameter comes from the configs list.
    """
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")

    with nsight.annotate("matmul"):
        _ = a @ b


def main() -> None:
    # notice no n parameter is passed, it is passed in the configs list instead
    result = benchmark_matmul_sizes()
    print(result.to_dataframe())
    print("✓ Benchmark complete! Check '02_parameter_sweep.png'")


if __name__ == "__main__":
    main()
