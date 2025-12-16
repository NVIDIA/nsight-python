# Copyright 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example 8: Collecting Multiple Metrics
=======================================

This example shows how to collect multiple metrics in a single profiling run.

New concepts:
- Using the `metrics` parameter to collect multiple metrics
- `@nsight.analyze.plot` decorator does NOT support multiple metrics now
"""

import torch

import nsight

sizes = [(2**i,) for i in range(11, 13)]


@nsight.analyze.kernel(
    configs=sizes,
    runs=5,
    # Collect both shared memory load and store SASS instructions
    metrics=[
        "smsp__sass_inst_executed_op_shared_ld.sum",
        "smsp__sass_inst_executed_op_shared_st.sum",
    ],
)
def analyze_shared_memory_ops(n: int) -> None:
    """Analyze both shared memory load and store SASS instructions
    for different kernels.

    Note: To evaluate multiple metrics, pass them as a sequence
    (list/tuple). All results are merged into one ProfileResults
    object, with the 'Metric' column indicating each specific metric.
    """

    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")
    c = torch.randn(2 * n, 2 * n, device="cuda")
    d = torch.randn(2 * n, 2 * n, device="cuda")

    with nsight.annotate("@-operator"):
        _ = a @ b

    with nsight.annotate("torch.matmul"):
        _ = torch.matmul(c, d)


def main() -> None:
    # Run analysis with multiple metrics
    results = analyze_shared_memory_ops()

    df = results.to_dataframe()
    print(df)

    unique_metrics = df["Metric"].unique()
    print(f"\n✓ Collected {len(unique_metrics)} metrics:")
    for metric in unique_metrics:
        print(f"  - {metric}")

    print("\n✓ Sample data:")
    print(df[["Annotation", "n", "Metric", "AvgValue"]].to_string(index=False))

    print("\n" + "=" * 60)
    print("IMPORTANT: @plot decorator limitation")
    print("=" * 60)
    print("When multiple metrics are collected:")
    print("  ✓ All metrics are collected in a single ProfileResults object")
    print("  ✓ DataFrame has 'Metric' column to distinguish them")
    print("  ✗ @nsight.analyze.plot decorator will RAISE AN ERROR")
    print("    Why? @plot can only visualize one metric at a time.")
    print("    Tip: Use separate @kernel functions for each metric or use")
    print("         'derive_metric' to compute custom values.")


if __name__ == "__main__":
    main()
