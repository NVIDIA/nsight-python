# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Visualization utilities for Nsight Python profiling and tensor difference analysis.

This module provides:
    - Plotting functions for profiling results with configurable layout and annotation.
"""
from collections.abc import Callable, Sequence
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nsight import exceptions, utils


def visualize(
    agg_df: str | pd.DataFrame,
    row_panels: Sequence[str] | None,
    col_panels: Sequence[str] | None,
    x_keys: Sequence[str] | None = None,
    print_data: bool = False,
    title: str = "",
    filename: str = "plot.png",
    ylabel: str = "",
    annotate_points: bool = True,
    show_avg: bool = True,
    plot_type: str = "line",
    plot_width: int = 6,
    plot_height: int = 4,
    show_geomean: bool = True,
    show_grid: bool = True,
    variant_fields: Sequence[Any] | None = None,
    variant_annotations: Sequence[Any] | None = None,
    plot_callback: Callable[[matplotlib.figure.Figure], None] | None = None,
) -> pd.DataFrame:
    """
    Plots profiling results using line or bar plots in a subplot grid.

    Args:
        agg_df: Aggregated profiling data or path to CSV file.
        row_panels: List of fields for whose unique values
            to create a new subplot along the vertical axis.
        col_panels: List of fields for whose unique values
            to create a new subplot along the horizontal axis.
        x_keys: List of fields to use for the x-axis. By
            default, we use all parameters of the decorated function except those
            specified in `row_panels` and `col_panels`.
        print_data: Whether to print aggregated profiling data to stdout.
        title: Main plot title.
        filename: Output filename for the saved plot.
        ylabel: Label for the y-axis (typically the metric name).
        annotate_points: Whether to annotate data points with values.
        show_avg: Whether to add an "Avg" column with average metric values.
        plot_type: Type of plot: "line" or "bar".
        show_geomean: Whether to show geometric mean values.
        show_grid: Whether to display grid lines on the plot.
        variant_fields: List of config fields to use as variant fields (lines).
        variant_annotations: List of annotated range names for which to apply variant splitting. The provided strings must each match one of the names defined using nsight.annotate.

    """
    if isinstance(agg_df, str):
        agg_df = pd.read_csv(agg_df)
    assert isinstance(
        agg_df, pd.DataFrame
    ), f"agg_df must be a pandas DataFrame or a CSV file path, not {type(agg_df)}"

    row_panels = row_panels or []
    col_panels = col_panels or []

    # --- Annotation Variants Expansion ---
    if variant_fields and variant_annotations:
        # Remove variant_fields from Configuration for all annotations
        config_exclude = set(variant_fields)
    else:
        config_exclude = set()

    # Build Configuration field excluding variant_fields
    annotation_idx = agg_df.columns.get_loc("AvgValue")
    func_fields = list(agg_df.columns[3:annotation_idx])
    subplot_fields = row_panels + col_panels  # type: ignore[operator]
    non_panel_fields = [
        field
        for field in func_fields
        if field not in subplot_fields and field not in config_exclude
    ]

    agg_df["Configuration"] = agg_df[non_panel_fields].apply(
        lambda row: ", ".join(
            f"{field_name}={value}" for field_name, value in row.items()
        ),
        axis=1,
    )

    # Expand variant annotations into separate lines, but keep x-ticks shared
    if variant_fields and variant_annotations:
        df = agg_df.copy()
        new_rows = []
        for annotation in variant_annotations:
            annotation_mask = df["Annotation"] == annotation
            if not annotation_mask.any():
                continue
            annotation_df = df[annotation_mask]
            # For each unique combination of variant_fields fields
            unique_combos = annotation_df[variant_fields].drop_duplicates()
            for _, combo in unique_combos.iterrows():
                combo_mask = (annotation_df[variant_fields] == combo.values).all(axis=1)
                combo_df = annotation_df[combo_mask].copy()
                # Create new annotation label
                variant_label = (
                    annotation
                    + " "
                    + ", ".join(
                        f"{k}={v}" for k, v in zip(variant_fields, combo.values)
                    )
                )
                combo_df["Annotation"] = variant_label
                new_rows.append(combo_df)
            # Remove the original annotation rows
            df = df[~annotation_mask]
        # Add all new variant rows
        if new_rows:
            df = pd.concat([df] + new_rows, ignore_index=True)
        agg_df = df

    # --- End Annotation Variants Expansion ---

    gpu_model = agg_df["GPU"].unique()[0]
    host = agg_df["Host"].unique()[0]
    hw_info_subtitle = f"{gpu_model}, {host}"
    title_with_hardware_info = f"{title}\n{hw_info_subtitle}"

    # Ensure that all fields in x_keys are present in non_panel_fields
    if x_keys:
        for field in x_keys:
            if field not in non_panel_fields:
                raise exceptions.ProfilerException(
                    f"Field '{field}' is not present in the DataFrame. "
                    f"Available fields: {non_panel_fields}"
                )

    unique_rows = (
        agg_df[row_panels].drop_duplicates()
        if row_panels
        else pd.DataFrame({"dummy": [0]})
    )
    unique_cols = (
        agg_df[col_panels].drop_duplicates()
        if col_panels
        else pd.DataFrame({"dummy": [0]})
    )

    nrows, ncols = max(len(unique_rows), 1), max(len(unique_cols), 1)
    # --- Ensure each subplot has a minimum size ---
    min_width_per_subplot = plot_width  # inches
    min_height_per_subplot = plot_height  # inches
    fig_width = max(min_width_per_subplot * ncols, 8)
    fig_height = max(min_height_per_subplot * nrows, 6)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        constrained_layout=True,  # Use constrained layout
    )

    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)
    if ncols == 1:
        axes = np.expand_dims(axes, axis=1)

    for row_idx, (_, row_values) in enumerate(unique_rows.iterrows()):
        for col_idx, (_, col_values) in enumerate(unique_cols.iterrows()):
            ax = axes[row_idx, col_idx]
            subplot_df = agg_df
            if row_panels:
                subplot_df = subplot_df[
                    (subplot_df[row_panels] == row_values).all(axis=1)
                ]
            if col_panels:
                subplot_df = subplot_df[
                    (subplot_df[col_panels] == col_values).all(axis=1)
                ]

            local_df = subplot_df.copy()

            # --- x_keys validation and Configuration building ---
            used_fields = (
                set(row_panels or [])
                | set(col_panels or [])
                | set(variant_fields or [])
            )
            if x_keys:
                overlap = used_fields & set(x_keys)
                if overlap:
                    raise ValueError(
                        f"x_keys cannot contain fields used in row_panels, col_panels, or variant_fields: {overlap}"
                    )
                config_fields = x_keys
            else:
                annotation_idx = local_df.columns.get_loc("AvgValue")
                func_fields = list(local_df.columns[3:annotation_idx])
                subplot_fields = row_panels + col_panels  # type: ignore[operator]
                config_exclude = set(variant_fields or [])
                config_fields = [
                    field
                    for field in func_fields
                    if field not in subplot_fields and field not in config_exclude
                ]
            local_df["Configuration"] = local_df[config_fields].apply(
                lambda row: ", ".join(
                    f"{field_name}={value}" for field_name, value in row.items()
                ),
                axis=1,
            )
            # --- End x_keys validation and Configuration building ---

            # --- Annotation Variants logic per subplot ---
            if variant_fields and variant_annotations:
                df = local_df.copy()
                new_rows = []
                for annotation in variant_annotations:
                    annotation_mask = df["Annotation"] == annotation
                    if not annotation_mask.any():
                        continue
                    annotation_df = df[annotation_mask]
                    unique_combos = annotation_df[variant_fields].drop_duplicates()
                    for _, combo in unique_combos.iterrows():
                        combo_mask = (
                            annotation_df[variant_fields] == combo.values
                        ).all(axis=1)
                        combo_df = annotation_df[combo_mask].copy()
                        variant_label = (
                            annotation
                            + " "
                            + ", ".join(
                                f"{k}={v}" for k, v in zip(variant_fields, combo.values)
                            )
                        )
                        combo_df["Annotation"] = variant_label
                        new_rows.append(combo_df)
                    df = df[~annotation_mask]
                if new_rows:
                    df = pd.concat([df] + new_rows, ignore_index=True)
                local_df = df
            # --- End Annotation Variants logic per subplot ---

            annotations = local_df["Annotation"].unique()
            nvidia_colors = [
                "#76B900",  # Green
                "#0070C5",  # Blue
                "#5C1682",  # Purple
                "#890C57",  # Red
                "#FAC200",  # Yellow
                "#008564",  # Dark Green
                "#FF5733",  # Orange
                "#C70039",  # Crimson
                "#900C3F",  # Dark Red
                "#581845",  # Dark Purple
                "#1F618D",  # Dark Blue
                "#28B463",  # Light Green
                "#F39C12",  # Bright Yellow
                "#D35400",  # Dark Orange
            ]

            unique_configs = local_df["Configuration"].dropna().unique()

            x_ticks = np.arange(len(unique_configs))
            n_annotations = len(annotations)
            width = (
                0.8 / n_annotations if n_annotations > 0 else 0.2
            )  # Adjust width for grouped bars
            avg_values = {}

            for i, (annotation, color) in enumerate(zip(annotations, nvidia_colors)):
                annotation_data = local_df[local_df["Annotation"] == annotation]
                # Keep annotation_data in original order (no sorting)

                # Compute valid average (for show_avg only)
                valid_values = annotation_data["AvgValue"].dropna()
                avg_value = valid_values.mean() if not valid_values.empty else np.nan
                avg_values[annotation] = avg_value

                # Map annotation_data to x positions (may be multiple points per x-tick)
                x_pos_map = {label: idx for idx, label in enumerate(unique_configs)}
                x_positions = annotation_data["Configuration"].map(x_pos_map)

                # Adjust x positions for grouped bars
                if plot_type == "bar" and n_annotations > 1:
                    # Center the group of bars around each x-tick
                    x_offset = (i - (n_annotations - 1) / 2) * width
                    x_positions = x_positions + x_offset

                if plot_type == "line":
                    ax.plot(
                        annotation_data["Configuration"].astype(str),
                        annotation_data["AvgValue"],
                        marker="o",
                        label=annotation,
                        color=color,
                    )
                elif plot_type == "bar":
                    ax.bar(
                        x_positions,
                        annotation_data["AvgValue"],
                        width=width,
                        label=annotation,
                        color=color,
                    )

                # Annotate each point with its value (formatted to 2 decimal places)
                if annotate_points:
                    for x_pos, y in zip(x_positions, annotation_data["AvgValue"]):
                        ax.text(
                            x_pos,
                            y + (0.02 if plot_type == "line" else 0.03),
                            f"{y:.2f}",
                            fontsize=8,
                            color=color,
                            ha="center",
                        )

            if show_avg:
                # Add vertical separation line for average values
                sep_index = len(unique_configs)
                ax.axvline(
                    x=sep_index - 0.5,
                    color="black",
                    linestyle="dashed",
                    linewidth=1,
                )

                for i, (annotation, color) in enumerate(
                    zip(annotations, nvidia_colors)
                ):
                    avg_value = avg_values.get(annotation, np.nan)
                    if not np.isnan(avg_value):  # Only plot if valid
                        avg_x_pos = sep_index + (i - (n_annotations - 1) / 2) * width
                        # Plot absolute value as bar on primary axis
                        ax.bar(avg_x_pos, avg_value, width=width, color=color, zorder=3)
                        # Always annotate the average values after the bar, with higher zorder
                        ax.text(
                            avg_x_pos,
                            avg_value + 0.02,
                            f"{avg_value:.2f}",
                            fontsize=9,
                            color=color,
                            ha="center",
                            fontweight="bold",
                            zorder=4,  # Ensure text is above the bar
                        )

            if show_geomean:
                # Add vertical separation line for geomean values
                sep_index = len(unique_configs) + (1 if show_avg else 0)
                ax.axvline(
                    x=sep_index - 0.5,
                    color="black",
                    linestyle="dashed",
                    linewidth=1,
                )

                for i, (annotation, color) in enumerate(
                    zip(annotations, nvidia_colors)
                ):
                    geomean = agg_df[agg_df["Annotation"] == annotation][
                        "Geomean"
                    ].iloc[0]
                    if not np.isnan(geomean):
                        geomean_x_pos = (
                            sep_index + (i - (n_annotations - 1) / 2) * width
                        )
                        ax.bar(
                            geomean_x_pos, geomean, width=width, color=color, zorder=3
                        )
                        ax.text(
                            geomean_x_pos,
                            geomean + 0.02,
                            f"{geomean:.2f}",
                            fontsize=9,
                            color=color,
                            ha="center",
                            fontweight="bold",
                            zorder=4,  # Ensure text is above the bar
                        )

            # Ensure all x-axis labels are strings
            x_labels = list(map(str, unique_configs))
            if show_avg:
                x_labels.append("Avg")
            if show_geomean:
                x_labels.append("Geomean")

            ax.set_xticks(np.arange(len(x_labels)))
            ax.set_xticklabels(
                x_labels,
                ha="right",  # Align to the right to prevent overlap
                rotation=45,  # Rotate for better readability
                fontsize=9,  # Reduce font size slightly
            )

            ax.set_ylim(0, max(agg_df["AvgValue"].max(skipna=True) * 1.1, 1))

            # Add grid
            if show_grid:
                ax.grid(True, linestyle="--", alpha=0.7)
                ax.set_axisbelow(True)  # Put grid behind the plot elements

            ylabel = ylabel or agg_df["Metric"].unique()[0]
            if col_idx == 0:
                ax.set_ylabel(f"{ylabel} (avg: {agg_df['NumRuns'].max()} runs)")

            # Generate combined subplot title with both row and col fields
            row_label = "\n".join(
                f"{field}={row_values[field]}" for field in row_panels
            )
            col_label = "\n".join(
                f"{field}={col_values[field]}" for field in col_panels
            )
            full_title = (
                f"{row_label}\n{col_label}"
                if row_label and col_label
                else row_label or col_label
            )
            # Count the number of linebreaks to adjust padding
            num_lines = full_title.count("\n") + 1 if full_title else 1
            ax.set_title(full_title, pad=18 + 6 * (num_lines - 1))

    # Use 'best' legend location for each axis
    for ax in axes.flat:
        ax.legend(
            title="Annotation",
            loc="best",  # Let matplotlib pick the best spot
            fontsize=9,
            title_fontsize=10,
        )

    # Set subplot titles with smaller font and more padding
    n_axes = len(axes.flat)
    for i, ax in enumerate(axes.flat):
        ax.set_title(ax.get_title(), fontsize=11, pad=18)
        # Only show x-axis labels and tick labels on the bottom row
        if nrows > 1:
            # If this axis is not in the last row, hide x labels
            if i < (nrows - 1) * ncols:
                ax.set_xlabel("")
                ax.set_xticklabels([])
        else:
            # If only one row, show all x labels
            pass

    # Set the main title, move it up a bit
    fig.suptitle(title_with_hardware_info, fontsize=14, fontweight="bold", y=1.08)

    if plot_callback:
        plot_callback(fig)

    # Save with tight bounding box to avoid clipping
    fig.savefig(filename, bbox_inches="tight")

    if print_data:
        agg_df["AvgValue"] = agg_df["AvgValue"].map("{:.4f}".format)
        print("Aggregated Data (Average value and Number of Runs):")
        print(agg_df.to_string(index=False))

    plt.close()
    return agg_df
