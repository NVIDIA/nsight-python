# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import abc
import dataclasses
import functools
import importlib.util
import inspect
import os
import threading
import time
import warnings
from collections.abc import Callable, Collection, Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from nsight import annotation, exceptions, thermovision, transformation, utils


def _sanitize_configs(
    func: Callable[..., Any],
    *args: Any,
    configs: Iterable[Any] | None = None,
    decorator_configs: Iterable[Any] | None = None,
    **kwargs: Any,
) -> list[tuple[Any, ...]]:
    """
    Sanitizes and validates configuration inputs for a profile-decorated function.

    This function ensures that the provided configurations are consistent and
    handles different cases of passing configurations.

    1. As regular args+kw - A single configuration -> the function arguments
    2. As configs=, an iterable of configurations at function call time
    3. As decorator_configs=, an iterable of configurations at decoration time
    4. No configs provided - For functions with no parameters, automatically creates [()]

    Args:
        func: The function being decorated.
        *args: Positional arguments that may contain configuration data.
        configs: An iterable of configurations provided at runtime.
        decorator_configs: An iterable of configurations provided
            at decoration time.
        **kwargs: Keyword arguments that may contain additional configuration data.

    Returns:
        A sanitized list of configurations.

    Raises:
        exceptions.ProfilerException: If no configurations are provided and the
            function has parameters, or if configurations are provided both at
            decoration time and runtime.
        AssertionError: If `configs` is not a list when provided.

    Notes:
        - If `args` are provided, `configs` and `decorator_configs` must be `None`.
        - If `configs` is provided, `decorator_configs` must be `None`, and vice versa.
        - For functions with no parameters, an empty config [()] is created automatically.
        - The function combines `args` and `kwargs` into a single list if `args` are provided.
        - The function assumes that `kwargs` keys are in the expected order when combining.
    """
    if len(args) > 0:
        # We do not expect any configs in this case
        assert configs is None and decorator_configs is None
        # kwargs not supported here yet
        if len(kwargs) != 0:
            raise exceptions.ProfilerException(
                f"Keyword arguments are not supported yet: {list(kwargs.keys())}"
            )

        configs = [list(args)]

    if configs is None:
        if decorator_configs is None:
            # Check if function takes no arguments
            sig = inspect.signature(func)
            expected_arg_count = len(sig.parameters)
            if expected_arg_count == 0:
                # For functions with no arguments, create a single empty config
                # This allows calling the function without requiring explicit configs
                configs = [()]
            else:
                raise exceptions.ProfilerException(
                    "You have provided no configs. Provide configs at decoration time or at runtime."
                )
        else:
            configs = decorator_configs

    else:
        if decorator_configs is not None:
            raise exceptions.ProfilerException(
                "You have provided configs at decoration time and at runtime. Provide configs at decoration time or at runtime."
            )

    assert isinstance(
        configs, Iterable
    ), f"configs must be an iterable, got {type(configs)}"

    if isinstance(configs, Collection):
        # Validate that all configs have the same number of arguments
        if len(configs) == 0:
            raise exceptions.ProfilerException("configs cannot be empty")

        # If function takes exactly one argument, allow scalar configs
        sig = inspect.signature(func)
        expected_arg_count = len(sig.parameters)
        if expected_arg_count == 1:
            normalized_configs: list[Sequence[Any]] = []
            for config in configs:
                if utils.is_scalar(config):
                    normalized_configs.append((config,))
                else:
                    normalized_configs.append(config)
            configs = normalized_configs

        config_lengths = [len(config) for config in configs]
        if not all(length == config_lengths[0] for length in config_lengths):
            raise exceptions.ProfilerException(
                f"All configs must have the same number of arguments. Found lengths: {config_lengths}"
            )
        first_config_arg_count = config_lengths[0]

        # Validate that the number of args matches the number of function parameters
        if first_config_arg_count != expected_arg_count:
            raise exceptions.ProfilerException(
                f"Configs have {first_config_arg_count} arguments, but function expects {expected_arg_count}"
            )

    return configs  # type: ignore[return-value]


def run_profile_session(
    func: Callable[..., None],
    configs: Iterable[Sequence[Any]],
    runs: int,
    output_progress: bool,
    output_detailed: bool,
    thermal_control: bool,
) -> None:

    if output_progress:
        print("")
        print("")

    if thermal_control:
        thermovision_initialized = thermovision.init()

    if isinstance(configs, Collection):
        total_configs = len(configs)
        total_runs = total_configs * runs
    else:
        total_configs = None
        total_runs = None

    curr_config = 0
    curr_run = 0
    total_time: float = 0
    bar_length = 100
    progress_time: float = 0

    # overwrite flag: we do not overwrite when output mode is detailed
    overwrite_output = not output_detailed
    show_return_type_warning = False
    config_lengths: list[int] = list()

    for c in configs:
        expected_arg_count = len(inspect.signature(func).parameters)

        # Handle scalar values
        if expected_arg_count == 1:
            if utils.is_scalar(c):
                c = (c,)

        # Check if func supports the input configs
        if expected_arg_count != len(c):
            raise exceptions.ProfilerException(
                f"Function '{func.__name__}' does not support the input configuration"
            )

        config_lengths.append(len(c))

        if config_lengths[0] != len(c):
            config_lengths.append(len(c))
            raise exceptions.ProfilerException(
                f"All configs must have the same number of arguments. Found lengths: {list(set(config_lengths))}"
            )

        curr_config += 1

        if output_progress:
            utils.print_config(total_configs, curr_config, c, overwrite_output)

        for i in range(runs):
            start_time = time.time()
            curr_run += 1
            if thermal_control:
                if thermovision_initialized:
                    thermovision.throttle_guard()

            # Clear active annotations before each run
            annotation.clear_active_annotations()

            # Run the function with the config
            result = func(*c)  # type: ignore[func-returns-value]
            if result is not None:
                show_return_type_warning = True

            elapsed_time = time.time() - start_time
            if curr_run > 1:
                total_time += elapsed_time
                avg_time_per_run = total_time / curr_run
            else:
                avg_time_per_run = elapsed_time  # Use first run's time only

            # Update time estimates every half second
            if time.time() - progress_time > 0.5:
                if output_progress and isinstance(configs, Collection):
                    utils.print_progress_bar(
                        total_runs,  # type: ignore[arg-type]
                        curr_run,
                        bar_length,
                        avg_time_per_run,
                        overwrite_output,
                    )
                progress_time = time.time()

    # Update progress bar at end so it shows 100%
    if output_progress and isinstance(configs, Collection):
        utils.print_progress_bar(
            total_runs,  # type: ignore[arg-type]
            curr_run,
            bar_length,
            avg_time_per_run,
            overwrite_output,
        )

    if show_return_type_warning:
        external_stacklevel = max(
            1, utils.find_external_stacklevel() - 1
        )  # We subtract 1 to exclude the stacklevel increase caused by the find_external_stack_level call
        warnings.warn(
            f"Function '{func.__name__}' returns a value, but the return value for a function which is decorated with nsight.analyze.kernel is ignored and instead the ProfileResults object is returned.",
            category=RuntimeWarning,
            stacklevel=external_stacklevel,
        )


@dataclasses.dataclass
class ProfileSettings:
    """
    Class to hold profile settings for Nsight Python.
    """

    configs: Iterable[Any] | None
    """
    An iterable  of configurations to run the function with.
    Each configuration can either be a sequence of arguments or
    if the decorated function takes only one argument, can also
    be a scalar value. If the configs are not provided at decoration
    time, they must be provided when calling the decorated function.
    """

    runs: int
    """Number of times each configuration should be executed."""

    output_progress: bool
    """
    Will display a progress bar on stdout during profiling
    """

    output_detailed: bool
    """
    Will display a progress bar, detailed output for each config along with the profiler logs
    """

    derive_metric: Callable[..., float] | None
    """
    A function to transform the collected metrics.
    This can be used to compute derived metrics like TFLOPs that cannot
    be captured by ncu directly. The function takes the metric values and
    the arguments of the profile-decorated function and returns the new
    metrics. See the examples for concrete use cases.
    """

    normalize_against: str | None
    """
    Annotation name to normalize metrics against.
    This is useful to compute relative metrics like speedup.
    """

    thermal_control: bool
    """
    Toggles whether to enable thermal control.
    """

    output_prefix: str | None
    """
    All intermediate profiler files are created with this prefix
    """

    output_csv: bool
    """
    Controls whether to output raw and processed profiling data to CSV files
    """


class ProfileResults:
    """
    Class to hold profile results for Nsight Python
    """

    def __init__(self, results: pd.DataFrame):
        """
        Initialize a ProfileResults object.

        Args:
            results: Processed profiling results.
        """
        self._results = results

    def to_dataframe(self) -> pd.DataFrame:
        """
        Returns the processed profiling data as a pandas DataFrame.

        This DataFrame contains aggregated statistics across multiple runs for each
        configuration and annotation combination. The data is equivalent to what is
        written to the ``processed_data-<function_name>-<run_id>.csv`` file when
        ``output_csv=True``.

        Returns:
            Processed profiling data with the following columns:

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
        return self._results


class NsightCollector(abc.ABC):
    @abc.abstractmethod
    def collect(
        self,
        func: Callable[..., Any],
        configs: Iterable[Any],
        settings: ProfileSettings,
    ) -> pd.DataFrame | None:
        """
        Collects profiling data for the given function and configurations.

        Args:
            func: The function to be profiled.
            configs: iterable of configurations for profiling.
            settings: Settings for profiling.

        Returns:
            Collected profiling data.
        """
        pass


class NsightProfiler:
    """
    A decorator class for profiling functions using Nsight Python's profiling tools.

    This class allows you to wrap a function with profiling capabilities,
    collecting performance data and saving the results to CSV files. It uses
    a collector to gather raw profiling data and processes it according to
    the provided settings.

    Attributes:
        settings: Configuration settings for profiling,
            including normalization, and other options.
        collector: The collector responsible for gathering
            raw profiling data.

    Methods:
        __call__(func):
            Wraps the given function with profiling logic. Collects raw
            profiling data, processes it, and saves the results to CSV files.
            Returns the processed data.
    """

    def __init__(self, settings: ProfileSettings, collector: NsightCollector):
        self.settings = settings
        self.collector = collector

    def __call__(
        self, func: Callable[..., None]
    ) -> Callable[..., ProfileResults | None]:
        func._nspy_ncu_run_id = 0  # type: ignore[attr-defined]

        @functools.wraps(func)
        def wrapper(
            *args: Any,
            configs: Iterable[Any] | None = None,
            **kwargs: Any,
        ) -> ProfileResults | None:

            tag = f"{func.__name__}-{func._nspy_ncu_run_id}"  # type: ignore[attr-defined]

            configs = _sanitize_configs(
                func,
                *args,
                configs=configs,
                decorator_configs=self.settings.configs,
                **kwargs,
            )

            raw_df = self.collector.collect(func, configs, self.settings)

            # Check if the function has a return type
            if raw_df is not None:

                processed: pd.DataFrame = transformation.aggregate_data(
                    raw_df,
                    func,
                    self.settings.normalize_against,
                    self.settings.output_progress,
                )

                # Save to CSV if enabled
                if self.settings.output_csv:
                    raw_csv_path = (
                        f"{self.settings.output_prefix}profiled_data-{tag}.csv"
                    )
                    processed_csv_path = (
                        f"{self.settings.output_prefix}processed_data-{tag}.csv"
                    )

                    raw_df.to_csv(
                        raw_csv_path,
                        index=False,
                    )
                    processed.to_csv(
                        processed_csv_path,
                        index=False,
                    )

                    if self.settings.output_progress:
                        print(
                            f"[NSIGHT-PYTHON] Refer to {raw_csv_path} for the raw profiling data"
                        )
                        print(
                            f"[NSIGHT-PYTHON] Refer to {processed_csv_path} for the processed profiling data"
                        )

                func._nspy_ncu_run_id += 1  # type: ignore[attr-defined]

                return ProfileResults(results=processed)

            return None

        return wrapper
