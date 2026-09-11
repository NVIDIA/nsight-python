# Copyright 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example 13: Custom Info Collectors
==================================

This example demonstrates how to use custom information collectors
to add additional columns to your profiling data using the decorator-based API.

New concepts:
- Using `@nsight.experimental.collect(scope=...)`
- Collecting information once, per configuration, per run, or per annotation
- Understanding collection scopes and aggregation behavior
"""

import subprocess
import time
from typing import Any

import torch
from cuda.core import system

import nsight


# Define custom collectors for "once" scope (collected once at start)
@nsight.experimental.collect(scope=nsight.CollectionScope.ONCE)
def driver_version() -> str:
    """Get the CUDA user-mode driver version."""
    try:
        return ".".join(str(part) for part in system.get_user_mode_driver_version())
    except Exception:
        return "unknown"


@nsight.experimental.collect(scope=nsight.CollectionScope.ONCE)
def cuda_version() -> str:
    """Get CUDA version from nvcc."""
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        # Parse output like: "Cuda compilation tools, release 12.9, V12.9.41"
        for line in result.stdout.split("\n"):
            if "release" in line and "V" in line:
                # Extract version after "V"
                version_part = line.split("V")[1].strip()
                # Take everything before first space/newline
                version = version_part.split()[0]
                return version
        return "unknown"
    except Exception:
        return "unknown"


# Define custom collectors for "run" scope (collected for every repeated run)
@nsight.experimental.collect(scope=nsight.CollectionScope.RUN)
def gpu_temp_c(*config_args: Any) -> int | None:  # pylint: disable=unused-argument
    """Get GPU temperature for the current configuration."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(result.stdout.strip().split("\n")[0])
    except Exception:
        return None


@nsight.experimental.collect(scope=nsight.CollectionScope.RUN)
def run_timestamp(*config_args: Any) -> str:  # pylint: disable=unused-argument
    """Get timestamp when each configuration run starts (formatted as HH:MM:SS)."""
    return time.strftime("%H:%M:%S", time.localtime())


# CONFIG collectors are evaluated once for each distinct configuration.
@nsight.experimental.collect(scope=nsight.CollectionScope.CONFIG)
def matrix_elements(*config_args: Any) -> int | None:
    """
    Example of a config-dependent collector.
    This demonstrates how you can compute derived values based on config parameters.
    """
    # config_args will contain (n, dtype) in this case
    if len(config_args) > 0:
        n: int = config_args[0]
        return n * n  # Total matrix elements
    return None


# Define custom collectors for "annotation" scope (collected for each annotation)
# Counter state for execution order tracking
_annotation_counter = {"count": 0}


@nsight.experimental.collect(scope=nsight.CollectionScope.ANNOTATION)
def exec_order(
    annotation_name: str, *config_args: Any
) -> int:  # pylint: disable=unused-argument
    """
    Counter that tracks the execution order of annotations.
    Increments with each annotation call to show execution sequence.
    """
    _annotation_counter["count"] += 1
    return _annotation_counter["count"]


@nsight.experimental.collect(scope=nsight.CollectionScope.ANNOTATION)
def power_draw_w(
    annotation_name: str, *config_args: Any
) -> float | None:  # pylint: disable=unused-argument
    """Get GPU power draw at the start of each annotation."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        # Parse "123.45 W" -> 123.45
        power_str = result.stdout.strip().split("\n")[0].replace(" W", "")
        return float(power_str)
    except Exception:
        return None


@nsight.experimental.collect(scope=nsight.CollectionScope.ANNOTATION)
def ann_info(annotation_name: str, *config_args: Any) -> str:
    """
    Example showing how to use both annotation name and config in a collector.
    """
    n = config_args[0] if config_args else 0
    return f"{annotation_name}_n{n}"


@nsight.analyze.plot("13_custom_info_collectors.png", print_data=True, plot_type="bar")
@nsight.analyze.kernel(
    runs=10,
    output_csv=True,
    verbosity=nsight.VerbosityLevel.DEBUG,
    info_collectors=[
        # "once" scope: collected once at the start of profiling
        driver_version,  # Column name: driver_version
        cuda_version,  # Column name: cuda_version
        # "run" scope: collected for every repeated run
        gpu_temp_c,  # Column name: gpu_temp_c
        run_timestamp,  # Column name: run_timestamp
        # "config" scope: collected once for each configuration
        matrix_elements,  # Column name: matrix_elements
        # "annotation" scope: collected for each annotation
        exec_order,  # Column name: exec_order
        power_draw_w,  # Column name: power_draw_w
        ann_info,  # Column name: ann_info
    ],
)
def benchmark_with_custom_info(n: int, dtype: torch.dtype) -> None:
    """
    Matrix multiplication benchmark with custom info collection.

    The resulting DataFrame will include:
    - Standard columns: Annotation, Value, Metric, Kernel, GPU, Host, etc.
    - Config parameters: n, dtype
    - Custom "once" columns: driver_version, cuda_version
    - Custom "run" columns: gpu_temp_c_Avg/Std/Min/Max, run_timestamp
    - Custom "config" columns: matrix_elements_Avg/Std/Min/Max
    - Custom "annotation" columns: exec_order_Avg/Std/Min/Max, power_draw_w_Avg/Std/Min/Max, ann_info
    """
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)

    with nsight.annotate("matmul"):
        _ = a @ b

    # Add another annotation to demonstrate annotation-scope collectors
    with nsight.annotate("add"):
        _ = a + b
        # Small delay to allow temperature/power to stabilize
        time.sleep(0.1)


def main() -> None:
    """Run the benchmark with multiple configurations."""
    configs = [
        (512, torch.float32),
        (1024, torch.float32),
        (2048, torch.float32),
        (1024, torch.float16),
        (2048, torch.float16),
    ]

    results = benchmark_with_custom_info(configs=configs)

    print("\n" + "=" * 80)
    print("Custom Info Collectors Example Complete!")
    print("=" * 80)
    print("\nThe DataFrame now includes custom columns:")
    print("\n  ONCE scope (constant across all runs):")
    print("    - driver_version")
    print("    - cuda_version")
    print("\n  CONFIG scope (collected once per configuration):")
    print("    - matrix_elements_Avg/Std/Min/Max (numeric)")
    print("\n  RUN scope (vary per repeated run, aggregated):")
    print("    - gpu_temp_c_Avg/Std/Min/Max (numeric)")
    print("    - run_timestamp (string, first value taken)")
    print("\n  ANNOTATION scope (vary per annotation):")
    print("    - exec_order_Avg/Std/Min/Max (numeric, shows execution order)")
    print("    - power_draw_w_Avg/Std/Min/Max (numeric, aggregated)")
    print("    - ann_info (string, first value taken)")
    print("\nNote: Run and annotation-scope collectors can vary across runs.")
    print("      Numeric collectors are aggregated (mean, std, min, max) like Value.")
    print("      Non-numeric collectors are kept as-is (first value taken).")
    print("\nCheck the printed DataFrame above to see all columns!")

    if results is not None:
        # Demonstrate accessing custom columns from all four scopes
        df = results.to_dataframe()

        # Show ONCE scope data (constant across all rows)
        print("\n" + "=" * 80)
        print("ONCE scope collectors (constant values):")
        print("=" * 80)
        if "driver_version" in df.columns:
            print(f"  Driver Version: {df['driver_version'].iloc[0]}")
        if "cuda_version" in df.columns:
            print(f"  CUDA Version:   {df['cuda_version'].iloc[0]}")

        # Show CONFIG and ANNOTATION scope data (varies, so aggregated)
        print("\n" + "=" * 80)
        print("CONFIG and ANNOTATION scope collectors (aggregated per run):")
        print("=" * 80)

        # Select columns to display
        display_cols = ["Annotation", "n", "dtype"]
        if "gpu_temp_c_Avg" in df.columns:
            display_cols.append("gpu_temp_c_Avg")
        if "run_timestamp" in df.columns:
            display_cols.append("run_timestamp")
        if "exec_order_Min" in df.columns:
            display_cols.append("exec_order_Min")
        if "power_draw_w_Max" in df.columns:
            display_cols.append("power_draw_w_Max")
        if "ann_info" in df.columns:
            display_cols.append("ann_info")

        print(df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
