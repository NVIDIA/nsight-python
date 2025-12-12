# Copyright 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example 6: Plot Customization
==============================

This example shows different visualization options.

New concepts:
- Using `plot_type` to change visualization style (line vs bar)
- Using `plot_callback` for advanced customization
- Setting titles and labels
"""

from typing import Any

import torch

import nsight

sizes = [(2**i,) for i in range(11, 14)]


def compute_tflops(time_ns: float, n: int) -> float:
    flops = 2 * n * n * n
    return flops / (time_ns / 1e9) / 1e12


# Example 1: Bar chart
@nsight.analyze.plot(
    filename="06_bar_chart.png",
    title="Matrix Multiplication Performance",
    ylabel="TFLOPs/s",
    plot_type="bar",  # Use bar chart instead of line plot
    annotate_points=True,
)
@nsight.analyze.kernel(configs=sizes, runs=10, derive_metric=compute_tflops)
def benchmark_bar_chart(n: int) -> None:
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")

    with nsight.annotate("matmul"):
        _ = a @ b


# Example 2: Custom plot callback for advanced styling
def custom_style(fig: Any) -> None:
    """
    Callback function to customize the plot appearance.
    """
    # Modify the figure
    fig.suptitle("Custom Styled Plot", fontsize=16, fontweight="bold")
    fig.set_figwidth(10)
    fig.set_figheight(6)

    # Modify the axes
    ax = fig.get_axes()[0]
    ax.set_ylabel("Performance (TFLOPs/s)", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Customize line appearance
    for line in ax.get_lines():
        line.set_linewidth(2)
        line.set_marker("o")
        line.set_markersize(8)


@nsight.analyze.plot(
    filename="06_custom_plot.png",
    plot_callback=custom_style,  # Apply custom styling
)
@nsight.analyze.kernel(configs=sizes, runs=10, derive_metric=compute_tflops)
def benchmark_custom_plot(n: int) -> None:
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")

    with nsight.annotate("matmul"):
        _ = a @ b


def main() -> None:
    print("Running bar chart example...")
    benchmark_bar_chart()
    print("✓ Bar chart saved to '06_bar_chart.png'")

    print("\nRunning custom plot example...")
    benchmark_custom_plot()
    print("✓ Custom plot saved to '06_custom_plot.png'")


if __name__ == "__main__":
    main()
