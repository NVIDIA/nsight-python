# Copyright 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example 9: Advanced Custom Metrics from Multiple Metrics
=========================================================

This example shows how to compute custom metrics from multiple metrics.

New concepts:
- Using `derive_metric` to compute custom values from multiple metrics
"""

import torch

import nsight

sizes = [(2**i,) for i in range(10, 13)]


def compute_avg_insts(
    ld_insts: int, st_insts: int, launch_sm_count: int, n: int
) -> float:
    """
    Compute average shared memory load/store instructions per SM.

    Custom metric function signature:
    - First several arguments: the measured metrics, must match the order
      of metrics in @kernel decorator
    - Remaining arguments: must match the decorated function's signature

    In this example:
    - ld_insts: Total shared memory load instructions
                (from smsp__inst_executed_pipe_lsu.shared_op_ld.sum metric)
    - st_insts: Total shared memory store instructions
                (from smsp__inst_executed_pipe_lsu.shared_op_st.sum metric)
    - launch_sm_count: Number of SMs that launched blocks
                (from launch__block_sm_count metric)
    - n: Matches the 'n' parameter from benchmark_avg_insts(n)

    Args:
        ld_insts: Total shared memory load instructions
        st_insts: Total shared memory store instructions
        launch_sm_count: Number of SMs that launched blocks
        n: Matrix size (n x n) - parameter from the decorated benchmark function

    Returns:
        Average shared memory load/store instructions per SM
    """
    insts_per_sm = (ld_insts + st_insts) / launch_sm_count
    return insts_per_sm


@nsight.analyze.plot(
    filename="09_advanced_metric_custom.png",
    ylabel="Average Shared Memory Load/Store Instructions per SM",  # Custom y-axis label
    annotate_points=True,  # Show values on the plot
)
@nsight.analyze.kernel(
    configs=sizes,
    runs=10,
    derive_metric=compute_avg_insts,  # Use custom metric
    metrics=[
        "smsp__sass_inst_executed_op_shared_ld.sum",
        "smsp__sass_inst_executed_op_shared_st.sum",
        "launch__sm_count",
    ],
)
def benchmark_avg_insts(n: int) -> None:
    """
    Benchmark matmul and display results.
    """
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")

    with nsight.annotate("matmul"):
        _ = a @ b


def main() -> None:
    result = benchmark_avg_insts()
    print(result.to_dataframe())
    print("✓ Avg Insts benchmark complete! Check '09_advanced_metric_custom.png'")


if __name__ == "__main__":
    main()
