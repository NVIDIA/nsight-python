# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Extraction utilities for analyzing NVIDIA Nsight Compute profiling data.

This module provides functionality to load `.ncu-rep` reports, extract performance data,
and transform it into structured pandas DataFrames for further analysis.

Functions:
    extract_ncu_action_data(action, metric):
        Extracts performance data for a specific kernel action from an NVIDIA Nsight Compute report.

    extract_df_from_report(metric, configs, iterations, func, derive_metric, ignore_kernel_list, verbose, combine_kernel_metrics=None):
        Processes the full NVIDIA Nsight Compute report and returns a pandas DataFrame containing performance metrics.
"""

import functools
import inspect
import socket
from collections.abc import Callable
from typing import Any, List, Tuple

import ncu_report
import pandas as pd

from nsight import exceptions, utils


def extract_ncu_action_data(action: Any, metric: str) -> utils.NCUActionData:
    """
    Extracts performance data from an NVIDIA Nsight Compute kernel action.

    Args:
        action: The NVIDIA Nsight Compute action object.
        metric: The metric name to extract from the action.

    Returns:
        A data container with extracted metric, clock rates, and GPU name.
    """
    if metric not in action.metric_names():
        error_message = exceptions.get_metric_error_message(
            metric, error_type=exceptions.MetricErrorType.UNSUPPORTED
        )
        raise exceptions.ProfilerException(error_message)

    return utils.NCUActionData(
        name=action.name(),
        value=(
            None if "dummy_kernel_failure" in action.name() else action[metric].value()
        ),
        compute_clock=action["device__attribute_clock_rate"].value(),
        memory_clock=action["device__attribute_memory_clock_rate"].value(),
        gpu=action["device__attribute_display_name"].value(),
    )


def extract_df_from_report(
    report_path: str,
    metric: str,
    configs: List[Tuple[Any, ...]],
    iterations: int,
    func: Callable[..., Any],
    derive_metric: Callable[..., Any] | None,
    ignore_kernel_list: List[str] | None,
    output_progress: bool,
    combine_kernel_metrics: Callable[[float, float], float] | None = None,
) -> pd.DataFrame:
    """
    Extracts and aggregates profiling results from an NVIDIA Nsight Compute report.

    Args:
        report_path: Path to the report file.
        metric: The NVIDIA Nsight Compute metric to extract.
        configs: Configuration settings used during profiling runs.
        iterations: Number of times each configuration was run.
        func: Function representing the kernel launch with parameter signature.
        derive_metric: Function to transform the raw metric value with config values.
        ignore_kernel_list: Kernel names to ignore in the analysis.
        combine_kernel_metrics: Function to merge multiple kernel metrics.
        verbose: Toggles the printing of extraction progress

    Returns:
        A DataFrame containing the extracted and transformed performance data.

    Raises:
        RuntimeError: If multiple kernels are detected per config without a combining function.
        exceptions.ProfilerException: If profiling results are missing or incomplete.
    """
    if output_progress:
        print("[NSIGHT-PYTHON] Loading profiled data")
    try:
        report = ncu_report.load_report(report_path)
    except FileNotFoundError:
        raise exceptions.ProfilerException(
            "No NVIDIA Nsight Compute report found. Please run nsight-python with `@nsight.analyze.kernel(output='verbose')`"
            "to identify the issue."
        )

    annotations: List[str] = []
    values: List[float | None] = []
    kernel_names: List[str] = []
    gpus: List[str] = []
    compute_clocks: List[int] = []
    memory_clocks: List[int] = []
    metrics: List[str] = []
    transformed_metrics: List[str | bool] = []
    hostnames: List[str] = []

    sig = inspect.signature(func)

    # Create a new array for each argument in the signature
    arg_arrays: dict[str, list[Any]] = {name: [] for name in sig.parameters.keys()}

    # Extract all profiling data
    if output_progress:
        print(f"Extracting profiling data")
    profiling_data: dict[str, list[utils.NCUActionData]] = {}
    for range_idx in range(report.num_ranges()):
        current_range = report.range_by_idx(range_idx)
        for action_idx in range(current_range.num_actions()):
            action = current_range.action_by_idx(action_idx)
            state = action.nvtx_state()

            for domain_idx in state.domains():
                domain = state.domain_by_id(domain_idx)

                # ignore actions not in the nsight-python nvtx domain
                if domain.name() != utils.NVTX_DOMAIN:
                    continue
                # ignore kernels in ignore_kernel_list
                if ignore_kernel_list and action.name() in ignore_kernel_list:
                    continue

                annotation = domain.push_pop_ranges()[0]
                data = extract_ncu_action_data(action, metric)

                if annotation not in profiling_data:
                    profiling_data[annotation] = []
                profiling_data[annotation].append(data)

    for annotation, annotation_data in profiling_data.items():
        if output_progress:
            print(f"Extracting {annotation} profiling data")

        configs_repeated = [config for config in configs for _ in range(iterations)]

        if len(annotation_data) == 0:
            raise RuntimeError("No kernels were profiled")
        if len(annotation_data) % len(configs_repeated) != 0:
            raise RuntimeError(
                "Expect same number of kernels per run. "
                f"Got average of {len(annotation_data) / len(configs_repeated)} per run"
            )
        num_kernels = len(annotation_data) // len(configs_repeated)

        if num_kernels > 1:
            if combine_kernel_metrics is None:
                raise RuntimeError(
                    (
                        f"More than one (total={num_kernels}) kernel is launched within the {annotation} annotation.\n"
                        "We expect one kernel per annotation.\n"
                        "Try one of the following solutions:\n"
                        "  - Use `replay_mode='range'` to profile the entire annotated range instead of individual kernels\n"
                        "  - Use `combine_kernel_metrics = lambda x, y: ...` to combine the metrics of multiple kernels\n"
                        "  - Add some of the kernels to the ignore_kernel_list\n"
                        "Kernels are:\n"
                        + "\n".join(sorted(set(x.name for x in annotation_data)))
                    )
                )

            assert (
                callable(combine_kernel_metrics)
                and combine_kernel_metrics.__code__.co_argcount == 2
            ), "Profiler error: combine_kernel_metrics must be a binary function"

        # rewrite annotation_data to combine the kernels
        action_data: list[utils.NCUActionData] = []
        for data_tuple in utils.batched(annotation_data, num_kernels):
            # Convert tuple to list for functools.reduce
            batch_list: list[utils.NCUActionData] = list(data_tuple)
            action_data.append(
                functools.reduce(
                    utils.NCUActionData.combine(combine_kernel_metrics), batch_list
                )
            )

        for conf, data in zip(configs_repeated, action_data):
            compute_clocks.append(data.compute_clock)
            memory_clocks.append(data.memory_clock)
            gpus.append(data.gpu)
            kernel_names.append(data.name)

            # evaluate the measured metric
            value = data.value
            if derive_metric is not None:
                derived_metric = None if value is None else derive_metric(value, *conf)
                value = derived_metric
                derive_metric_name = derive_metric.__name__
                transformed_metrics.append(derive_metric_name)
            else:
                transformed_metrics.append(False)

            values.append(value)

            # gather remaining required data
            annotations.append(annotation)
            metrics.append(metric)
            hostnames.append(socket.gethostname())
            # Add a field for every config argument
            bound_args = sig.bind(*conf)
            for name, val in bound_args.arguments.items():
                arg_arrays[name].append(val)

    # Create the DataFrame with the initial columns
    df_data = {
        "Annotation": annotations,
        "Value": values,
        "Metric": metrics,
        "Transformed": transformed_metrics,
        "Kernel": kernel_names,
        "GPU": gpus,
        "Host": hostnames,
        "ComputeClock": compute_clocks,
        "MemoryClock": memory_clocks,
    }

    # Add each array in arg_arrays to the DataFrame
    for arg_name, arg_values in arg_arrays.items():
        df_data[arg_name] = arg_values

    return pd.DataFrame(df_data)
