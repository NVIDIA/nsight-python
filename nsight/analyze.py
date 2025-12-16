# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import functools
import os
import tempfile
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, overload

import matplotlib
import matplotlib.figure
import numpy as np

import nsight.collection as collection
import nsight.visualization as visualization


# Overload 1: When used without parentheses: @kernel
@overload
def kernel(
    _func: Callable[..., None],
) -> Callable[..., collection.core.ProfileResults]: ...


# Overload 2: When used with parentheses: @kernel() or @kernel(args)
@overload
def kernel(
    _func: None = None,
    *,
    configs: Iterable[Any] | None = None,
    runs: int = 1,
    derive_metric: Callable[..., float] | None = None,
    normalize_against: str | None = None,
    output: Literal["quiet", "progress", "verbose"] = "progress",
    metrics: Sequence[str] = ["gpu__time_duration.sum"],
    ignore_kernel_list: Sequence[str] | None = None,
    clock_control: Literal["base", "none"] = "none",
    cache_control: Literal["all", "none"] = "all",
    replay_mode: Literal["kernel", "range"] = "kernel",
    thermal_control: bool = True,
    combine_kernel_metrics: Callable[[float, float], float] | None = None,
    output_prefix: str | None = None,
    output_csv: bool = False,
) -> Callable[[Callable[..., None]], Callable[..., collection.core.ProfileResults]]: ...


# Implementation
def kernel(
    _func: Callable[..., None] | None = None,
    *,
    configs: Iterable[Any] | None = None,
    runs: int = 1,
    derive_metric: Callable[..., float] | None = None,
    normalize_against: str | None = None,
    output: Literal["quiet", "progress", "verbose"] = "progress",
    metrics: Sequence[str] = ["gpu__time_duration.sum"],
    ignore_kernel_list: Sequence[str] | None = None,
    clock_control: Literal["base", "none"] = "none",
    cache_control: Literal["all", "none"] = "all",
    replay_mode: Literal["kernel", "range"] = "kernel",
    thermal_control: bool = True,
    combine_kernel_metrics: Callable[[float, float], float] | None = None,
    output_prefix: str | None = None,
    output_csv: bool = False,
) -> (
    Callable[..., collection.core.ProfileResults]
    | Callable[[Callable[..., None]], Callable[..., collection.core.ProfileResults]]
):
    """
    A decorator that collects profiling data using NVIDIA Nsight Compute.

    Can be used with or without parentheses:
        - ``@nsight.analyze.kernel`` (no parentheses)
        - ``@nsight.analyze.kernel()`` (empty parentheses)
        - ``@nsight.analyze.kernel(configs=..., runs=10)`` (with arguments)

    The decorator returns a wrapped version of your function with the following signature::

        def wrapped_function(*args, configs=None, **kwargs) -> ProfileResults

    Where:
        - ``*args``: Original function arguments (when providing a single config)
        - ``configs``: Optional iterable of configurations (overrides decorator-time configs)
        - ``**kwargs``: Original function keyword arguments
        - Returns ``ProfileResults`` object containing profiling data


    Parameters:
        configs: An iterable of configurations to run the function with. Each configuration can be either:

            - A sequence of arguments to pass to the decorated function:

                   configs = [ [1, 2], [3, 4], ]

            - If the decorated function takes only one argument, it can be a scalar value:

                   configs = [1, 2, 3, 4]

            If the configs are not provided at decoration time, they must be provided when calling the decorated function.
        runs:  Number of times each configuration should be executed.
        derive_metric:
            A function to transform the collected metrics.
            This can be used to compute derived metrics like TFLOPs that cannot
            be captured by ncu directly. The function takes the metric values and
            the arguments of the profile-decorated function and returns the new
            metric. The parameter order requirements for the custom function:

            - First several arguments: Must exactly match the order of metrics declared in the @kernel decorator. These arguments will receive the actual measured values of those metrics.
            - Remaining arguments: Must exactly match the signature of the decorated function. In other words, the original function's parameters are passed in order.

            See the examples for concrete use cases.
        normalize_against:
            Annotation name to normalize metrics against.
            This is useful to compute relative metrics like speedup.
        metrics: The metrics to collect. By default, kernel runtimes in nanoseconds
            are collected. Default: ``["gpu__time_duration.sum"]``. To see the available
            metrics on your system, use the command: ``ncu --query-metrics``.
        ignore_kernel_list:
            List of kernel names to ignore. If you call a library within an annotated range context, you might not have precise control over which and how many kernels are being launched.
            If some of these kernels should be ignored in the profile, their names can be provided in this parameter. Default: ``None``
        combine_kernel_metrics: By default, Nsight Python
            expects one kernel launch per annotation. In case an annotated region launches
            multiple kernels, instead of failing the profiling run, you can specify
            how to summarize the collected metrics into a single number. For example,
            if we profile runtime and want to sum the times of all kernels we can specify
            ``combine_kernel_metrics = lambda x, y: x + y``. The function should take
            two arguments and return a single value. Default: ``None``.
        clock_control: Control the behavior of the GPU clocks during profiling. Allowed values:

            - ``"base"``: GPC and memory clocks are locked to their respective base frequency during profiling. This has no impact on thermal throttling. Note that actual clocks might still vary, depending on the level of driver support for this feature. As an alternative, use nvidia-smi to lock the clocks externally and set this option to ``"none"``.
            - ``"none"``: No GPC or memory frequencies are changed during profiling.

            Default: ``"none"``
        cache_control: Control the behavior of the GPU caches during profiling. Allowed values:

            - ``"all"``: All GPU caches are flushed before each kernel replay iteration during profiling. While metric values in the execution environment of the application might be slightly different without invalidating the caches, this mode offers the most reproducible metric results across the replay passes and also across multiple runs of the target application.
            - ``"none"``: No GPU caches are flushed during profiling. This can improve performance and better replicates the application behavior if only a single kernel replay pass is necessary for metric collection. However, some metric results will vary depending on prior GPU work, and between replay iterations. This can lead to inconsistent and out-of-bounds metric values.

            Default: ``"all"``

        replay_mode: Mechanism used for replaying a kernel launch multiple times to collect selected metrics. Allowed values:

            - ``"kernel"``:  Replay individual kernel launches  during the execution of the application.
            - ``"range"``: Replay range of  kernel launches during the execution of the application. Ranges are defined using nsight.annotate.

            Default: ``"kernel"``

        thermal_control : Toggles whether to enable thermal control. Default: ``True``
        output: Controls the verbosity level of the output.

            - ``"quiet"``: Suppresses all output.
            - ``"progress"``: Shows a progress bar along with details about profiling and data extraction progress.
            - ``"verbose"``: Displays the progress bar, configuration-specific logs, and profiler logs.

        output_prefix: When specified, all intermediate profiler files are created with this prefix.
            For example, if `output_prefix="/home/user/run1_"`, the profiler will generate:

            - /home/user/run1_ncu-output-<name_of_decorated_function>-<run_id>.log
            - /home/user/run1_ncu-output-<name_of_decorated_function>-<run_id>.ncu-rep
            - /home/user/run1_processed_data-<name_of_decorated_function>-<run_id>.csv
            - /home/user/run1_profiled_data-<name_of_decorated_function>-<run_id>.csv

            Where ``<run_id>`` is a counter that increments each time the decorated function is called
            within the same Python process (0, 1, 2, ...). This allows calling the same decorated function
            multiple times without overwriting previous results.

            if ``None``, the intermediate profiler files are created in a directory under <TEMP_DIR> prefixed with nspy. <TEMP_DIR> is the system's temporary directory (`$TMPDIR` or `/tmp` on Linux, `%TEMP%` on Windows).

        output_csv: Controls whether to dump raw and processed profiling data to CSV files. Default: ``False``.
            When enabled, two CSV files are generated:

            **Raw Data CSV** (``profiled_data-<function_name>-<run_id>.csv``): Contains unprocessed profiling data with one row per run per configuration. Columns include:

                - ``Annotation``: Name of the annotated region being profiled
                - ``Value``: Raw metric values collected by the profiler
                - ``Metric``: The metrics being collected (e.g., ``gpu__time_duration.sum``)
                - ``Transformed``: Name of the function used to transform the metrics (specified via ``derive_metric``), or ``False`` if no transformation was applied. For lambda functions, this shows ``"<lambda>"``
                - ``Kernel``: Name of the GPU kernel(s) launched
                - ``GPU``: GPU device name
                - ``Host``: Host machine name
                - ``ComputeClock``: GPU compute clock frequency during profiling
                - ``MemoryClock``: GPU memory clock frequency during profiling
                - ``<param_name>``: One column for each parameter of the decorated function

            **Processed Data CSV** (``processed_data-<function_name>-<run_id>.csv``): Contains aggregated statistics across multiple runs. Columns include:

                - ``Annotation``: Name of the annotated region being profiled
                - ``<param_name>``: One column for each parameter of the decorated function
                - ``AvgValue``: Average metric values across all runs
                - ``StdDev``: Standard deviation of the metrics across runs
                - ``MinValue``: Minimum metric values observed
                - ``MaxValue``: Maximum metric values observed
                - ``NumRuns``: Number of runs used for aggregation
                - ``CI95_Lower``: Lower bound of the 95% confidence interval
                - ``CI95_Upper``: Upper bound of the 95% confidence interval
                - ``RelativeStdDevPct``: Standard deviation as a percentage of the mean
                - ``StableMeasurement``: Boolean indicating if the measurement is stable (low variance). The measurement is stable if ``RelativeStdDevPct`` < 2 % .
                - ``Metric``: The metrics being collected
                - ``Transformed``: Name of the function used to transform the metrics (specified via ``derive_metric``), or ``False`` if no transformation was applied. For lambda functions, this shows ``"<lambda>"``
                - ``Kernel``: Name of the GPU kernel(s) launched
                - ``GPU``: GPU device name
                - ``Host``: Host machine name
                - ``ComputeClock``: GPU compute clock frequency
                - ``MemoryClock``: GPU memory clock frequency
    """
    # Strip whitespace
    metrics = [m.strip() for m in metrics]

    def _create_profiler() -> collection.core.NsightProfiler:
        """Helper to create the profiler with the given settings."""
        if output not in ("quiet", "progress", "verbose"):
            raise ValueError("output must be 'quiet', 'progress' or 'verbose'")

        output_progress = output == "progress" or output == "verbose"
        output_detailed = output == "verbose"

        # Create the output paths needed for the ncu report, ncu logs and the CSVs
        prefix = output_prefix
        if "NSPY_NCU_PROFILE" not in os.environ:
            if prefix is None:
                prefix = tempfile.mkdtemp(prefix="nspy_")
                prefix = os.path.join(
                    prefix, ""
                )  # Adds a trailing forward/backward slash
            else:
                os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)

        settings = collection.core.ProfileSettings(
            configs=configs,
            runs=runs,
            output_progress=output_progress,
            output_detailed=output_detailed,
            derive_metric=derive_metric,
            normalize_against=normalize_against,
            thermal_control=thermal_control,
            output_prefix=prefix,
            output_csv=output_csv,
        )
        ncu = collection.ncu.NCUCollector(
            metrics=metrics,
            ignore_kernel_list=ignore_kernel_list,
            combine_kernel_metrics=combine_kernel_metrics,
            clock_control=clock_control,
            cache_control=cache_control,
            replay_mode=replay_mode,
        )
        return collection.core.NsightProfiler(settings, ncu)

    # Support both @kernel and @kernel() syntax
    if _func is None:
        # Called with parentheses: @kernel() or @kernel(args)
        return _create_profiler()  # type: ignore[return-value]
    else:
        # Called without parentheses: @kernel
        # _func is the decorated function, so we need to apply the profiler to it
        profiler = _create_profiler()
        return profiler(_func)  # type: ignore[return-value]


def _validate_metric(result: collection.core.ProfileResults) -> None:
    """
    Check if ProfileResults contains only a single metric.

    Args:
        result: ProfileResults object

    Raises:
        ValueError: If multiple metrics are detected
    """
    df = result.to_dataframe()

    # Check for multiple metrics in "Metric" column
    unique_metrics = df["Metric"].unique()
    if len(unique_metrics) > 1:
        raise ValueError(
            f"Cannot visualize {len(unique_metrics)} > 1 metrics with the "
            "@nsight.analyze.plot decorator."
        )


def plot(
    filename: str = "plot.png",
    *,
    title: str = "Nsight Analyze Kernel Plot Results",
    ylabel: str | None = None,
    annotate_points: bool = False,
    show_aggregate: str | None = None,
    plot_type: str = "line",
    plot_width: int = 6,
    plot_height: int = 4,
    row_panels: Sequence[str] | None = None,
    col_panels: Sequence[str] | None = None,
    x_keys: Sequence[str] | None = None,
    print_data: bool = False,
    variant_fields: Sequence[str] | None = None,
    variant_annotations: Sequence[str] | None = None,
    plot_callback: Callable[[matplotlib.figure.Figure], None] | None = None,
) -> Callable[
    [Callable[..., collection.core.ProfileResults]],
    Callable[..., collection.core.ProfileResults],
]:
    """
    A decorator that plots the result of a profile-decorated function.
    This decorator is intended to be only used on functions that have been
    decorated with `@nsight.analyze.kernel`.

    The decorator returns a wrapped version of your function that maintains the same
    signature as the underlying ``@nsight.analyze.kernel`` decorated function::

        def wrapped_function(*args, configs=None, **kwargs) -> ProfileResults

    The function returns ``ProfileResults`` and generates a plot as a side effect.

    Example usage::

        @nsight.analyze.plot(title="My Plot")
        @nsight.analyze.kernel
        def my_func(...):

    Args:
        filename: Filename to save the plot. Default: ``'plot'``
        title: Title for the plot. Default: ``'Nsight Analyze Kernel Plot Results'``
        ylabel: Label for the y-axis in the generated plot.
            Default: ``f'{metric} (avg: {runs} runs)'``
        annotate_points: If True, annotate the points with
            their numeric value in the plot.
            Default: ``False``
        show_aggregate: If “avg”, show the average value in the plot. If “geomean”, show the geometric mean value in the plot.
            Default: None
        plot_type: Type of plot to generate. Options are
            'line' or 'bar'. Default: ``'line'``
        plot_width: Width of the plot in inches. Default: ``6``
        plot_height: Height of the plot in inches. Default: ``4``
        row_panels: Enables generating subplots along
            the horizontal axis for each unique values of the listed function parameters.
            The provided strings must each match one argument of the
            nsight.analyze.kernel-decorated function. Default: ``None``
        col_panels: Enables generating subplots along
            the vertical axis for each unique values of the listed function parameters.
            The provided strings must each match one argument of the
            nsight.analyze.kernel-decorated function. Default: ``None``
        x_keys: List of fields to use for the x-axis. By
            default, we use all parameters of the decorated function except those
            specified in `row_panels` and `col_panels`.
        print_data: If True, print the data used for plotting.
            Default: ``False``
        variant_fields: List of config fields to use as variant fields (lines).
        variant_annotations: List of annotated range names for which to apply variant splitting. The provided strings must each match one of the names defined using nsight.annotate.
    """
    show_avg = show_aggregate == "avg"
    show_geomean = show_aggregate == "geomean"

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)

            if "NSPY_NCU_PROFILE" not in os.environ:
                # Check for multiple metrics or complex data structures
                _validate_metric(result)

                visualization.visualize(
                    result.to_dataframe(),
                    row_panels=row_panels,
                    col_panels=col_panels,
                    x_keys=x_keys,
                    print_data=print_data,
                    title=title,
                    filename=filename,
                    ylabel=ylabel or "",
                    annotate_points=annotate_points,
                    show_avg=show_avg,
                    show_geomean=show_geomean,
                    plot_type=plot_type,
                    plot_width=plot_width,
                    plot_height=plot_height,
                    variant_fields=variant_fields,
                    variant_annotations=variant_annotations,
                    plot_callback=plot_callback,
                )
            return result

        return wrapper

    return decorator


# ------------------------------------------------------------------------------
# nsight.analyze.ignore_failures context manager
# For ignoring errors in warmup runs outside nsight.annotate
# ------------------------------------------------------------------------------
@contextlib.contextmanager
def ignore_failures() -> Any:
    """
    Context manager that ignores errors in a code block.

    Useful when you want failures in the block to be suppressed so they
    do not propagate and cause the decorated function to fail.
    """
    try:
        yield
    except Exception:
        pass
