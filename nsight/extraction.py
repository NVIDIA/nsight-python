# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Extraction utilities for analyzing NVIDIA Nsight Compute profiling data.

This module provides functionality to load `.ncu-rep` reports, extract performance data,
and transform it into structured pandas DataFrames for further analysis.

Functions:
    extract_ncu_action_data(action, metrics):
        Extracts performance data for a specific kernel action from an NVIDIA Nsight Compute report.

    extract_df_from_report(report_path, metrics, configs, iterations, func, derive_metric, ignore_kernel_list, output_progress, combine_kernel_metrics=None):
        Processes the full NVIDIA Nsight Compute report and returns a pandas DataFrame containing performance metrics.
"""

import functools
import inspect
import socket
from collections.abc import Callable, Sequence
from typing import Any, List, Tuple, TypeAlias

import ncu_report
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from nsight import exceptions, utils
from nsight.utils import is_scalar

DerivedValue: TypeAlias = float | int | None
DerivedValueDict: TypeAlias = dict[str, DerivedValue]


def extract_ncu_action_data(action: Any, metrics: Sequence[str]) -> utils.NCUActionData:
    """
    Extracts performance data from an NVIDIA Nsight Compute kernel action.

    Args:
        action: The NVIDIA Nsight Compute action object.
        metrics: The metric names to extract from the action.

    Returns:
        A data container with extracted metrics, clock rates, and GPU name.
    """
    for metric in metrics:
        if metric not in action.metric_names():
            error_message = exceptions.get_metrics_error_message(
                metric, error_type=exceptions.MetricErrorType.INVALID
            )
            raise exceptions.ProfilerException(error_message)

    # Extract values for all metrics.
    failure = "dummy_kernel_failure" in action.name()
    all_values = (
        None if failure else np.array([action[metric].value() for metric in metrics])
    )

    return utils.NCUActionData(
        name=action.name(),
        values=all_values,
        compute_clock=action["device__attribute_clock_rate"].value(),
        memory_clock=action["device__attribute_memory_clock_rate"].value(),
        gpu=action["device__attribute_display_name"].value(),
    )


def extract_df_from_report(
    report_path: str,
    metrics: Sequence[str],
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
        metrics: The NVIDIA Nsight Compute metrics to extract.
        configs: Configuration settings used during profiling runs.
        iterations: Number of times each configuration was run.
        func: Function representing the kernel launch with parameter signature.
        derive_metric: Function to transform the raw metric values with config values.
        ignore_kernel_list: Kernel names to ignore in the analysis.
        combine_kernel_metrics: Function to merge multiple kernel metrics.
        verbose: Toggles the printing of extraction progress.

    Returns:
        A DataFrame containing the extracted and transformed performance data.

    Raises:
        RuntimeError: If multiple kernels are detected per config without a combining function.
        exceptions.ProfilerException: If profiling results are missing or incomplete.
    """
    if output_progress:
        print("[NSIGHT-PYTHON] Loading profiled data")
    try:
        report: ncu_report.IContext = ncu_report.load_report(report_path)
    except FileNotFoundError:
        raise exceptions.ProfilerException(
            "No NVIDIA Nsight Compute report found. Please run nsight-python with `@nsight.analyze.kernel(output='verbose')`"
            "to identify the issue."
        )

    annotations: List[str] = []
    all_values: List[Tuple[Any, ...] | None] = []
    all_transformed_values: List[
        List[DerivedValue] | DerivedValue | NDArray[Any] | None
    ] = []
    kernel_names: List[str] = []
    gpus: List[str] = []
    compute_clocks: List[int] = []
    memory_clocks: List[int] = []
    all_metrics: List[Tuple[str, ...]] = []
    all_transformed_metrics: List[List[str] | str | bool] = []
    hostnames: List[str] = []

    sig = inspect.signature(func)

    # Create a new array for each argument in the signature
    arg_arrays: dict[str, list[Any]] = {name: [] for name in sig.parameters.keys()}

    # Extract all profiling data
    if output_progress:
        print(f"Extracting profiling data")
    profiling_data: dict[str, list[utils.NCUActionData]] = {}
    for range_idx in range(report.num_ranges()):
        current_range: ncu_report.IRange = report.range_by_idx(range_idx)
        for action_idx in range(current_range.num_actions()):
            action: ncu_report.IAction = current_range.action_by_idx(action_idx)
            state: ncu_report.INvtxState = action.nvtx_state()

            for domain_idx in state.domains():
                domain: ncu_report.INvtxDomainInfo = state.domain_by_id(domain_idx)

                # ignore actions not in the nsight-python nvtx domain
                if domain.name() != utils.NVTX_DOMAIN:
                    continue
                # ignore kernels in ignore_kernel_list
                if ignore_kernel_list and action.name() in ignore_kernel_list:
                    continue

                annotation: str = domain.push_pop_ranges()[0]
                data = extract_ncu_action_data(action, metrics)

                if annotation not in profiling_data:
                    profiling_data[annotation] = []
                profiling_data[annotation].append(data)

    for annotation, annotation_data in profiling_data.items():
        if output_progress:
            print(f"Extracting {annotation} profiling data")

        configs_repeated = [
            (config,) if is_scalar(config) else config
            for config in configs
            for _ in range(iterations)
        ]

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

            # evaluate the measured metrics
            values = data.values
            if derive_metric is not None:
                if not callable(derive_metric):
                    raise TypeError("derive_metric must be a callable function")

                if values is not None:
                    derive_metric_params = inspect.signature(derive_metric).parameters
                    has_varargs: bool = any(
                        p.kind == inspect.Parameter.VAR_POSITIONAL
                        for p in derive_metric_params.values()
                    )
                    actual_params = None if has_varargs else len(derive_metric_params)
                    # If there are varargs, skip the check
                    if actual_params is not None:
                        expected_params = len(values) + len(conf)
                        if actual_params != expected_params:
                            raise ValueError(
                                f"derive_metric expects {expected_params} parameters "
                                f"({len(values)} metric values + {len(conf)} configs), "
                                f"but has {actual_params} parameters"
                            )

                derived_metric: DerivedValueDict | DerivedValue = (
                    None if values is None else derive_metric(*values, *conf)
                )
                if isinstance(derived_metric, dict):
                    # If the derived metric is a dict, then we have multiple metrics
                    # and use the keys of the dict as metric names.
                    metric_names, metric_values = zip(
                        *[(k, v) for k, v in derived_metric.items()]
                    )
                    all_transformed_values.append(list(metric_values))
                    all_transformed_metrics.append(list(metric_names))
                else:
                    # If the derived metric is a scalar, then we have a single metric
                    # and use the name of the function as the metric name.
                    all_transformed_values.append(derived_metric)
                    all_transformed_metrics.append(derive_metric.__name__)

            # gather remaining required data
            annotations.append(annotation)
            all_values.append(tuple(values) if values is not None else None)
            all_metrics.append(tuple(metrics))
            hostnames.append(socket.gethostname())
            # Add a field for every config argument
            bound_args = sig.bind(*conf)
            for name, val in bound_args.arguments.items():
                arg_arrays[name].append(val)

    # Create the DataFrame with the initial columns
    df_data = {
        "Annotation": annotations,
        "Value": all_values,
        "Metric": all_metrics,
        "Kernel": kernel_names,
        "GPU": gpus,
        "Host": hostnames,
        "ComputeClock": compute_clocks,
        "MemoryClock": memory_clocks,
    }

    # Add each array in arg_arrays to the DataFrame
    for arg_name, arg_values in arg_arrays.items():
        df_data[arg_name] = arg_values

    # Explode the dataframe
    df = pd.DataFrame(df_data).apply(pd.Series.explode).reset_index(drop=True)

    if derive_metric is not None:
        transformed_df_data = {
            "Annotation": annotations,
            "Value": all_transformed_values,
            "Metric": all_transformed_metrics,
            "Kernel": kernel_names,
            "GPU": gpus,
            "Host": hostnames,
            "ComputeClock": compute_clocks,
            "MemoryClock": memory_clocks,
        }

        for arg_name, arg_values in arg_arrays.items():
            transformed_df_data[arg_name] = arg_values

        transformed_df = (
            pd.DataFrame(transformed_df_data)
            .apply(pd.Series.explode)
            .reset_index(drop=True)
        )

        # Concat the two dataframes
        df = pd.concat([df, transformed_df], ignore_index=True)

    return df
