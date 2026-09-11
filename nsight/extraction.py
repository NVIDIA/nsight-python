# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Extraction utilities for analyzing NVIDIA Nsight Compute profiling data.

This module provides functionality to load `.ncu-rep` reports, extract performance data,
and transform it into structured pandas DataFrames for further analysis.

Functions:
    extract_ncu_action_data(action, metrics):
        Extracts performance data for a specific kernel action from an NVIDIA Nsight Compute report.

    extract_df_from_report(report_path, metrics, configs, iterations, func, derive_metric, ignore_kernel_list, verbosity, combine_kernel_metrics=None):
        Processes the full NVIDIA Nsight Compute report and returns a pandas DataFrame containing performance metrics.
"""

import functools
import inspect
import socket
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any, List, Tuple, TypeAlias

import ncu_report
import numpy as np
import pandas as pd

from nsight import exceptions, utils
from nsight.utils import VerbosityLevel, is_scalar

DerivedValue: TypeAlias = float | int | None
DerivedValueWithUnit: TypeAlias = tuple[DerivedValue, str]
DerivedValueDict: TypeAlias = Mapping[str, DerivedValue | DerivedValueWithUnit]

# Warning message template for missing units in derived metrics
DERIVED_METRIC_MISSING_UNIT_WARNING = (
    "Derived metric '{}' does not have a unit specified. "
    "Return a tuple (value, unit) instead of just the value. "
    "np.nan will be added to the 'Unit' column in the dataframe."
)


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
    all_units = [action[metric].unit() for metric in metrics]

    return utils.NCUActionData(
        name=action.name(),
        values=all_values,
        compute_clock=action["device__attribute_clock_rate"].value(),
        memory_clock=action["device__attribute_memory_clock_rate"].value(),
        gpu=action["device__attribute_display_name"].value(),
        units=all_units,
    )


def extract_df_from_report(
    report_path: str,
    metrics: Sequence[str],
    configs: List[Tuple[Any, ...]],
    iterations: int,
    func: Callable[..., Any],
    derive_metric: Callable[..., Any] | None,
    ignore_kernel_list: List[str] | None,
    verbosity: VerbosityLevel,
    combine_kernel_metrics: Callable[[float, float], float] | None = None,
    info_collectors_list: List[Tuple[str, Callable[..., Any], str]] | None = None,
    config_scope_columns: List[str] | None = None,
    annotation_scope_columns: List[str] | None = None,
    info_prefix: str | None = None,
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
        verbosity: Controls display of extraction progress.

    Returns:
        A DataFrame containing the extracted and transformed performance data.

    Raises:
        RuntimeError: If multiple kernels are detected per config without a combining function.
        exceptions.ProfilerException: If profiling results are missing or incomplete.
    """
    if verbosity >= VerbosityLevel.INFO:
        print("[NSIGHT-PYTHON] Loading profiled data")
    try:
        report: ncu_report.IContext = ncu_report.load_report(report_path)
    except FileNotFoundError:
        raise exceptions.ProfilerException(
            "No NVIDIA Nsight Compute report found. Please run nsight-python with `@nsight.analyze.kernel(verbosity=nsight.VerbosityLevel.DEBUG)`"
            "to identify the issue."
        )

    annotations: List[str] = []
    all_values: List[Tuple[Any, ...] | None] = []
    all_transformed_values: List[
        List[DerivedValue] | DerivedValue | np.typing.NDArray[Any] | None
    ] = []
    all_transformed_values_units: list[List[str | float] | str | float] = []
    kernel_names: List[str] = []
    gpus: List[str] = []
    compute_clocks: List[int] = []
    memory_clocks: List[int] = []
    all_metrics: List[Tuple[str, ...]] = []
    all_transformed_metrics: List[List[str] | str | bool] = []
    hostnames: List[str] = []
    units: List[List[str]] = []

    sig = inspect.signature(func)

    # Create a new array for each regular argument in the signature (exclude *args/**kwargs)
    arg_arrays: dict[str, list[Any]] = {
        name: []
        for name, p in sig.parameters.items()
        if p.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }

    # Load collected info from the pickle file if it exists
    collected_info_per_run = []
    custom_info_names = set()
    if info_collectors_list:
        import os

        # Use the explicitly-provided prefix when available so the read path
        # matches the collection-session write path exactly.
        # Fall back to reconstructing from the report directory for direct callers.
        if info_prefix is None:
            report_dir = os.path.dirname(report_path)
            info_prefix = os.path.join(report_dir, "") if report_dir else ""

        # Column names come from the collector metadata (authoritative), not from
        # the first run's collected data. Deriving from run 0 dropped any
        # annotation-scope collector whose annotation was not entered on run 0.
        custom_info_names = {name for name, _callback, _scope in info_collectors_list}

        info_file = f"{info_prefix}collected_info.pkl"
        if os.path.exists(info_file):
            import pickle

            with open(info_file, "rb") as f:
                collected_info_per_run = pickle.load(f)
            if not isinstance(collected_info_per_run, list):
                warnings.warn(
                    f"Collected info file {info_file} is malformed "
                    f"(expected a list, got {type(collected_info_per_run).__name__}); "
                    "ignoring custom info.",
                    category=RuntimeWarning,
                    stacklevel=2,
                )
                collected_info_per_run = []
            elif verbosity >= VerbosityLevel.INFO:
                print(f"[NSIGHT-PYTHON] Loaded collected info from {info_file}")
        else:
            if verbosity >= VerbosityLevel.INFO:
                print(
                    f"[NSIGHT-PYTHON] Warning: No collected info file found at {info_file}"
                )

    # Create arrays for custom info collectors
    custom_info_arrays: dict[str, list[Any]] = {name: [] for name in custom_info_names}

    # Extract all profiling data
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

                annotation: str = domain.start_end_ranges()[0]
                data = extract_ncu_action_data(action, metrics)

                if annotation not in profiling_data:
                    profiling_data[annotation] = []
                profiling_data[annotation].append(data)

    for annotation, annotation_data in profiling_data.items():
        if verbosity >= VerbosityLevel.INFO:
            print(f"[NSIGHT-PYTHON] Extracting {annotation} profiling data")

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

        for idx, (conf, data) in enumerate(
            zip(configs_repeated, action_data, strict=True)
        ):
            compute_clocks.append(data.compute_clock)
            memory_clocks.append(data.memory_clock)
            gpus.append(data.gpu)
            kernel_names.append(data.name)
            units.append(data.units)

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

                # Store function name for warning messages
                derive_metric_func_name = derive_metric.__name__
                derived_metric: (
                    DerivedValueWithUnit | DerivedValueDict | DerivedValue
                ) = (None if values is None else derive_metric(*values, *conf))
                if isinstance(derived_metric, Mapping):
                    # If the derived metric is a dict, then we have multiple metrics
                    # and use the keys of the dict as metric names.

                    metric_names: list[str] = list()
                    metric_values: list[DerivedValue] = list()
                    unit_names: list[str | float] = list()

                    for metric_name, metric_value in derived_metric.items():
                        metric_names.append(metric_name)

                        if isinstance(metric_value, tuple) and len(metric_value) >= 2:
                            metric_values.append(metric_value[0])
                            unit_names.append(metric_value[1])
                        else:
                            # It is a scalar value
                            metric_values.append(metric_value)
                            unit_names.append(np.nan)
                            warnings.warn(
                                DERIVED_METRIC_MISSING_UNIT_WARNING.format(metric_name)
                            )

                    all_transformed_values.append(list(metric_values))
                    all_transformed_metrics.append(list(metric_names))
                    all_transformed_values_units.append(list(unit_names))

                else:
                    if isinstance(derived_metric, tuple):
                        derived_metric_value = derived_metric[0]
                        derived_metric_unit = derived_metric[1]
                        all_transformed_values.append(derived_metric_value)
                        all_transformed_values_units.append(derived_metric_unit)

                    else:
                        warnings.warn(
                            DERIVED_METRIC_MISSING_UNIT_WARNING.format(
                                derive_metric_func_name
                            )
                        )
                        all_transformed_values.append(derived_metric)
                        all_transformed_values_units.append(np.nan)
                    all_transformed_metrics.append(derive_metric_func_name)

            # gather remaining required data
            annotations.append(annotation)
            all_values.append(tuple(values) if values is not None else None)
            all_metrics.append(tuple(metrics))
            hostnames.append(socket.gethostname())
            # Add a field for every config argument
            config_iter = iter(conf)
            for name, param in sig.parameters.items():
                if param.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    continue
                arg_arrays[name].append(next(config_iter))

            # Add custom collected information
            if (
                collected_info_per_run
                and idx < len(collected_info_per_run)
                and isinstance(collected_info_per_run[idx], dict)
            ):
                run_info = collected_info_per_run[idx]
                # Add once info. Guard against pickle keys that are not in the
                # metadata-derived custom_info_names (e.g. a stale collected_info
                # .pkl written by a different set of collectors); such keys have
                # no column and are ignored rather than raising KeyError.
                for name, value in run_info.get("once_info", {}).items():
                    if name in custom_info_arrays:
                        custom_info_arrays[name].append(value)
                # Add config info
                for name, value in run_info.get("config_info", {}).items():
                    if name in custom_info_arrays:
                        custom_info_arrays[name].append(value)
                # Add annotation info - match by annotation name
                annotation_info_list = run_info.get("annotation_info", [])
                annotation_data_for_this = {}
                for ann_data in annotation_info_list:
                    if ann_data.get("annotation") == annotation:
                        annotation_data_for_this = ann_data.get("data", {})
                        break
                for name in custom_info_names:
                    if name in annotation_data_for_this:
                        custom_info_arrays[name].append(annotation_data_for_this[name])
                    elif name not in run_info.get(
                        "once_info", {}
                    ) and name not in run_info.get("config_info", {}):
                        # This is an annotation-scope collector that wasn't collected for this annotation
                        custom_info_arrays[name].append(None)
            else:
                # No collected info available for this run, append None
                for name in custom_info_names:
                    custom_info_arrays[name].append(None)

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
        "Unit": units,
    }

    # Add custom info collector data BEFORE function parameters
    # This is important because transformation.py expects function params to be the LAST columns
    for collector_name, collector_values in custom_info_arrays.items():
        df_data[collector_name] = collector_values

    # Add each array in arg_arrays to the DataFrame (function parameters LAST)
    for arg_name, arg_values in arg_arrays.items():
        df_data[arg_name] = arg_values

    # Explode only Value, Metric and Unit columns (which contain tuples of per-metric data).
    # Other columns (including function args) may also contain tuples that should NOT be exploded.
    df = (
        pd.DataFrame(df_data)
        .explode(["Value", "Metric", "Unit"])
        .reset_index(drop=True)
    )

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
            "Unit": all_transformed_values_units,
        }

        # Custom-info columns must be added BEFORE the function-parameter columns
        # (mirroring df_data above) so transformation.py still treats the function
        # params as the LAST columns. Without this, derived-metric rows would get
        # NaN/None for every collector column after the pd.concat below. The
        # arrays are populated once per base row in the same loop, so they align
        # row-for-row with the transformed rows.
        for collector_name, collector_values in custom_info_arrays.items():
            transformed_df_data[collector_name] = collector_values

        for arg_name, arg_values in arg_arrays.items():
            transformed_df_data[arg_name] = arg_values

        transformed_df = (
            pd.DataFrame(transformed_df_data)
            .explode(["Value", "Metric", "Unit"])
            .reset_index(drop=True)
        )

        # Concat the two dataframes
        df = pd.concat([df, transformed_df], ignore_index=True)

    # Mark config-scope and annotation-scope columns for aggregation during transformation
    if config_scope_columns:
        df.attrs["config_scope_columns"] = config_scope_columns
    if annotation_scope_columns:
        df.attrs["annotation_scope_columns"] = annotation_scope_columns

    return df
