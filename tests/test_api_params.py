# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
from typing import Any, List

import torch

import nsight

# powers of two, 1k - 4k
sizes = [(2**i,) for i in range(10, 13)]


def get_app_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test with command line options to test parameters for nsight.annotate(), nsight.analyze.kernel() and nsight.analyze.plot()."
    )
    # nsight.analyze.kernel() parameters
    # TBD no command line arguments yet for: configs, derive_metric, ignore_kernel_list, combine_kernel_metrics
    parser.add_argument(
        "--metrics", "-m", default=["dram__bytes.sum.per_second"], help="Metric name"
    )
    parser.add_argument("--runs", "-r", type=int, default=10, help="Number of runs")
    parser.add_argument("--replay-mode", "-p", default="kernel", help="Replay mode")
    parser.add_argument(
        "--normalize-against", "-n", default=None, help="Value to normalize against"
    )
    parser.add_argument(
        "--clock-control", "-c", default="none", help="Clock control value"
    )
    parser.add_argument(
        "--cache-control", "-a", default="all", help="Cache control value"
    )
    parser.add_argument(
        "--thermal-control",
        "-t",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable thermal control",
    )
    parser.add_argument(
        "--output", "-o", default="progress", help="Output verbosity level"
    )
    parser.add_argument(
        "--output-prefix",
        "-op",
        default=None,
        help="Select the output prefix of the intermediate profiler files",
    )
    # nsight.analyze.plot() parameters
    # TBD no command line arguments yet for: row_panels, col_panels, x_keys, annotate_points, show_aggregate
    parser.add_argument("--plot-title", "-l", default="test", help="Plot title")
    parser.add_argument(
        "--plot-filename", "-f", default="params_test1.png", help="Plot filename"
    )
    parser.add_argument("--plot-type", "-y", default="line", help="Plot type")
    parser.add_argument(
        "--plot-print-data",
        "-i",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable printing plot data",
    )

    # nsight.annotate() parameters
    parser.add_argument("--annotate1", "-1", default="matmul", help="Annotation name 1")
    parser.add_argument("--annotate2", "-2", default="einsum", help="Annotation name 2")
    parser.add_argument("--annotate3", "-3", default="linear", help="Annotation name 3")

    args = parser.parse_args()

    return args


def main(argv: List[str]) -> None:
    args = get_app_args()

    for key, value in vars(args).items():
        print(f"{key}: {value}")

    @nsight.annotate(args.annotate2)
    def einsum(a: torch.Tensor, b: torch.Tensor) -> Any:
        return torch.einsum("ij,jk->ik", a, b)

    @nsight.analyze.plot(
        title=args.plot_title,
        filename=args.plot_filename,
        plot_type=args.plot_type,
        print_data=args.plot_print_data,
    )
    @nsight.analyze.kernel(
        configs=sizes,
        runs=args.runs,
        metrics=args.metrics,
        replay_mode=args.replay_mode,
        normalize_against=args.normalize_against,
        clock_control=args.clock_control,
        cache_control=args.cache_control,
        thermal_control=args.thermal_control,
        output=args.output,
        output_prefix=args.output_prefix,
    )
    def run_benchmark(n: int) -> None:
        a = torch.randn(n, n, device="cuda")
        b = torch.randn(n, n, device="cuda")

        _ = a @ b  # workaround for cuInit() issue

        with nsight.annotate(args.annotate1):
            _ = a @ b

        einsum(a, b)

        with nsight.annotate(args.annotate3):
            _ = torch.nn.functional.linear(a, b)

    run_benchmark()


if __name__ == "__main__":
    main(sys.argv[1:])
