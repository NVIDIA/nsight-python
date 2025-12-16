# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Data transformation utilities for Nsight Python profiling output.

This module contains functions that process raw profiling results, aggregate metrics,
normalize them, and prepare the data for visualization or further statistical analysis.
"""

import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd


def aggregate_data(
    df: pd.DataFrame,
    func: Callable[..., Any],
    normalize_against: str | None,
    output_progress: bool,
) -> pd.DataFrame:
    """
    Groups and aggregates profiling data by configuration and annotation.

    Args:
        df: The raw profiling results.
        func: Function representing kernel configuration parameters.
        normalize_against: Name of the annotation to normalize against.
        output_progress: Toggles the display of data processing logs

    Returns:
        Aggregated DataFrame and the (possibly normalized) metric name.
    """
    if output_progress:
        print("[NSIGHT-PYTHON] Processing profiled data")

    # Get the number of arguments in the signature of func
    num_args = len(inspect.signature(func).parameters)

    # Get the last N fields of the dataframe where N is the number of arguments
    # Note: When num_args=0, we need an empty list (not all columns via [-0:])
    func_fields = df.columns[-num_args:].tolist() if num_args > 0 else []

    # Function to convert non-sortable columns to tuples or strings
    def convert_non_sortable_columns(dframe: pd.DataFrame) -> pd.DataFrame:
        for col in dframe.columns:
            try:
                # Try sorting the column to check if it's sortable.
                sorted(dframe[col].dropna())
            except (TypeError, ValueError):
                # If sorting fails, convert the column to string
                dframe[col] = dframe[col].astype(str)
        return dframe

    # Convert non-sortable columns before grouping
    df = convert_non_sortable_columns(df)

    # Preserve original order by adding an index column
    df = df.reset_index(drop=True)
    df["_original_order"] = df.index

    # Build named aggregation dict for static fields
    named_aggs = {
        "AvgValue": ("Value", "mean"),
        "StdDev": ("Value", "std"),
        "MinValue": ("Value", "min"),
        "MaxValue": ("Value", "max"),
        "NumRuns": ("Value", "count"),
        "_original_order": (
            "_original_order",
            "min",
        ),  # Use min to preserve first occurrence
    }

    # The columns to aggregate except for the function parameters
    groupby_columns = ["Annotation", "Metric", "Transformed"]

    # Add assertion-based unique selection for remaining fields
    remaining_fields = [
        col
        for col in df.columns
        if col not in [*groupby_columns, "Value", "_original_order"] + func_fields
    ]

    for col in remaining_fields:
        if col == "Kernel":
            named_aggs[col] = (col, "first")
        else:
            named_aggs[col] = (  # type: ignore[assignment]
                col,
                (
                    lambda colname: lambda x: (
                        x.unique()[0]
                        if len(x.unique()) == 1
                        else (_ for _ in ()).throw(
                            AssertionError(
                                f"Column '{colname}' has multiple values in group: {x.unique()}"
                            )
                        )
                    )
                )(col),
            )

    # Apply aggregation with named aggregation
    groupby_df = df.groupby(groupby_columns + func_fields)
    agg_df = groupby_df.agg(**named_aggs).reset_index()

    # Compute 95% confidence intervals
    agg_df["CI95_Lower"] = agg_df["AvgValue"] - 1.96 * (
        agg_df["StdDev"] / np.sqrt(agg_df["NumRuns"])
    )
    agg_df["CI95_Upper"] = agg_df["AvgValue"] + 1.96 * (
        agg_df["StdDev"] / np.sqrt(agg_df["NumRuns"])
    )

    # Compute relative standard deviation as a percentage
    agg_df["RelativeStdDevPct"] = (agg_df["StdDev"] / agg_df["AvgValue"]) * 100

    # Flag measurements as stable if relative stddev is less than 2%
    agg_df["StableMeasurement"] = agg_df["RelativeStdDevPct"] < 2.0

    # Flatten the multi-index columns
    agg_df.columns = [col if isinstance(col, str) else col[0] for col in agg_df.columns]

    # Sort by original order to preserve user-provided configuration order
    agg_df = agg_df.sort_values("_original_order").reset_index(drop=True)
    agg_df = agg_df.drop("_original_order", axis=1)  # Remove the helper column

    do_normalize = normalize_against is not None
    if do_normalize:
        assert (
            normalize_against in agg_df["Annotation"].values
        ), f"Annotation '{normalize_against}' not found in data."

        # Columns of normalization dataframe to merge on
        merge_on = func_fields + ["Metric", "Transformed"]

        # Create a DataFrame to hold the normalization values
        normalization_df = agg_df[agg_df["Annotation"] == normalize_against][
            merge_on + ["AvgValue"]
        ]
        normalization_df = normalization_df.rename(
            columns={"AvgValue": "NormalizationValue"}
        )

        # Merge with the original DataFrame to apply normalization
        agg_df = pd.merge(agg_df, normalization_df, on=merge_on)

        # Normalize the AvgValue by the values of the normalization annotation
        agg_df["AvgValue"] = agg_df["NormalizationValue"] / agg_df["AvgValue"]

        # Update the metric name to reflect the normalization
        agg_df["Metric"] = (
            agg_df["Metric"].astype(str) + f" relative to {normalize_against}"
        )

    # Calculate the geometric mean of the AvgValue column
    agg_df["Geomean"] = agg_df.groupby(groupby_columns)["AvgValue"].transform(
        lambda x: (
            np.exp(np.mean(np.log(pd.to_numeric(x.dropna(), errors="coerce"))))
            if not x.dropna().empty
            else np.nan
        )
    )

    return agg_df
