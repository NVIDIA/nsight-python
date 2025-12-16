# Copyright 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example 11: Controlling CSV Output Files
=========================================

This example shows how to control CSV file generation.

New concepts:
- Using `output_csv` parameter to enable/disable CSV file generation
- Using `output_prefix` to specify output file location and naming
"""

import os

import pandas as pd
import torch

import nsight

# Get current directory for output
current_dir = os.path.dirname(os.path.abspath(__file__))
output_prefix = f"{current_dir}/example10_"


# Matrix sizes to benchmark
sizes = [(2**i,) for i in range(10, 13)]


@nsight.analyze.kernel(
    configs=sizes,
    runs=3,
    output_prefix=output_prefix,
    output_csv=True,  # Enable CSV file generation
    metrics=[
        "smsp__sass_inst_executed_op_shared_ld.sum",
        "smsp__sass_inst_executed_op_shared_st.sum",
    ],
)
def analyze_memory_ops_with_csv(n: int) -> None:
    """
    Analyze memory operations with CSV output enabled.

    When output_csv=True, two CSV files are generated:
    1. {prefix}processed_data-<name_of_decorated_function>-<run_id>.csv - Raw profiled data
    2. {prefix}profiled_data-<name_of_decorated_function>-<run_id>.csv - Processed/aggregated data

    Args:
        n: Matrix size (n x n)
    """
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")

    with nsight.annotate("matmul-operator"):
        _ = a @ b

    with nsight.annotate("torch-matmul"):
        _ = torch.matmul(a, b)


def print_full_dataframe(
    df: pd.DataFrame, max_rows: int = 20, max_col_width: int = 100
) -> None:
    """
    Print DataFrame without truncation.

    Args:
        df: DataFrame to print
        max_rows: Maximum number of rows to display (None for all rows)
        max_col_width: Maximum column width (None for no limit)
    """
    # Save current display options
    original_options = {
        "display.max_rows": pd.get_option("display.max_rows"),
        "display.max_columns": pd.get_option("display.max_columns"),
        "display.max_colwidth": pd.get_option("display.max_colwidth"),
        "display.width": pd.get_option("display.width"),
        "display.expand_frame_repr": pd.get_option("display.expand_frame_repr"),
    }

    try:
        # Set display options for full output
        pd.set_option("display.max_rows", max_rows if max_rows else None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_colwidth", max_col_width if max_col_width else None)
        pd.set_option("display.width", None)
        pd.set_option("display.expand_frame_repr", False)

        print(df.to_string())

    finally:
        # Restore original options
        for option, value in original_options.items():
            pd.set_option(option, value)


def read_and_display_csv_files() -> None:
    """Read and display the generated CSV files."""

    # Find CSV files
    csv_files = []
    for file in os.listdir(current_dir):
        if file.startswith("example10_") and file.endswith(".csv"):
            csv_files.append(os.path.join(current_dir, file))

    for file_path in sorted(csv_files):
        file_name = os.path.basename(file_path)
        print(f"\nFile: {file_name}")
        print("-" * (len(file_name) + 6))

        # Read CSV file
        try:
            df = pd.read_csv(file_path)

            # Display only columns related to metrics/values
            value_cols = [
                col
                for col in df.columns
                if "Value" in col or "Metric" in col or "Annotation" in col
            ]
            # print(df[value_cols].head())
            # Show full DataFrame without truncation
            print_full_dataframe(df[value_cols])
        except Exception as e:
            print(f"Error reading {file_name}: {e}")


def main() -> None:
    # Clean up any previous output files
    for old_file in os.listdir(current_dir):
        if old_file.startswith("example10_") and old_file.endswith(
            (".csv", ".ncu-rep", ".log")
        ):
            os.remove(os.path.join(current_dir, old_file))

    # Run the analysis with CSV output
    result = analyze_memory_ops_with_csv()
    print(result.to_dataframe())

    # Read and display generated CSV files
    read_and_display_csv_files()


if __name__ == "__main__":
    main()
