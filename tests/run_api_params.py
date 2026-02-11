# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CLI tool for manually testing API parameters and scenarios.

This tool imports TEST_SCENARIOS from test_api_params.py and provides
a command-line interface for QA and manual testing.

Usage:
    python run_api_params.py --list-scenarios
    python run_api_params.py --scenario inconsistent_kernel_counts
"""

import argparse
import sys
from typing import Any, Dict, List, Optional

import torch

# Import scenarios and helper from pytest file
from test_api_params import TEST_SCENARIOS, create_benchmark, sizes

import nsight


def list_scenarios() -> None:
    """
    Print all available test scenarios.
    """
    print("Available Test Scenarios:")
    print("=" * 50)

    for name, config in TEST_SCENARIOS.items():
        print(f"\n{name}:")
        print(f"   {config['description']}")
        print(f"   Expected Error: {config['expected']}")
        print(f"   Command: python run_api_params.py --scenario {name}")


def resolve_param(
    scenario: Optional[Dict[str, Any]], param_name: str, cli_value: Any
) -> Any:
    """Resolve parameter value: scenario override > CLI value > built-in default."""
    if scenario and param_name in scenario:
        return scenario[param_name]
    return cli_value


def resolve_all_params(
    scenario: Optional[Dict[str, Any]], args: argparse.Namespace
) -> Dict[str, Any]:
    """Resolve all parameters using priority: scenario > cli arg > default."""
    resolved_params: Dict[str, Any] = {}
    # Provide safe defaults for missing attributes
    defaults: Dict[str, Any] = {
        "decorator_configs": sizes,
        "runtime_configs": None,
        "benchmark_type": "default",
    }
    for param in [
        "decorator_configs",
        "runtime_configs",
        "metrics",
        "runs",
        "replay_mode",
        "normalize_against",
        "clock_control",
        "cache_control",
        "thermal_mode",
        "output",
        "output_prefix",
        "benchmark_type",
        "plot_title",
        "plot_filename",
        "plot_type",
        "plot_print_data",
        "annotate1",
        "annotate2",
        "annotate3",
    ]:
        value = getattr(args, param, defaults.get(param, None))
        resolved_params[param] = resolve_param(scenario, param, value)
    return resolved_params


def get_app_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test with command line options to test parameters for nsight.annotate(), nsight.analyze.kernel() and nsight.analyze.plot()."
    )

    # Add scenario selection
    parser.add_argument(
        "--scenario",
        "-sc",
        choices=TEST_SCENARIOS.keys(),
        default=None,
        help="Select test scenario to run",
    )

    parser.add_argument(
        "--list-scenarios",
        "-ls",
        action="store_true",
        help="List all available test scenarios",
    )

    # nsight.analyze.kernel() parameters
    # TBD no command line arguments yet for: derive_metric, ignore_kernel_list, combine_kernel_metrics
    parser.add_argument(
        "--metrics",
        "-m",
        nargs="+",
        default=["dram__bytes.sum.per_second"],
        help="Metric names (can specify multiple)",
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

    if args.list_scenarios:
        list_scenarios()
        return

    # Get scenario if any
    scenario = TEST_SCENARIOS.get(args.scenario, None)

    params = resolve_all_params(scenario, args)

    # Print resolved parameters
    print(f"\nParameters:")
    print("=" * 50)
    for key, value in params.items():
        print(f"   {key}: {value}")

    if scenario:
        print("=" * 50)
        print(f"\n# Running Scenario: {args.scenario}")
        print(f"# Description: {scenario['description']}")
        print(f"# Expected Error: {scenario['expected']}")

    # Build kernel decorator parameters
    kernel_kwargs = {
        "runs": params["runs"],
        "metrics": params["metrics"],
        "replay_mode": params["replay_mode"],
        "normalize_against": params["normalize_against"],
        "clock_control": params["clock_control"],
        "cache_control": params["cache_control"],
        "thermal_mode": params["thermal_mode"],
        "output": params["output"],
        "output_prefix": params["output_prefix"],
    }

    if params["decorator_configs"] is not None:
        kernel_kwargs["configs"] = params["decorator_configs"]

    benchmark_type = params["benchmark_type"]
    run_benchmark = create_benchmark(benchmark_type, kernel_kwargs, params)

    # Apply plot decorator once to the selected benchmark
    run_benchmark = nsight.analyze.plot(
        title=params["plot_title"],
        filename=params["plot_filename"],
        plot_type=params["plot_type"],
        print_data=params["plot_print_data"],
    )(run_benchmark)

    # Execute the test
    try:
        if params["runtime_configs"] is not None:
            run_benchmark(configs=params["runtime_configs"])
        else:
            run_benchmark()

    except Exception as e:
        print("=" * 50)
        print(f"\nError: {e}")


if __name__ == "__main__":
    main(sys.argv[1:])
