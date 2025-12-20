# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Collection utilities for profiling Nsight Python runs using NVIDIA Nsight Compute (ncu).

This module contains logic for launching NVIDIA Nsight Compute with appropriate settings.
NCU is instructed to profile specific code sections marked by NVTX ranges - the
Nsight Python annotations.
"""

import os
import subprocess
import sys
from collections.abc import Callable, Collection, Iterable, Sequence
from typing import Any, List, Literal, cast

import pandas as pd
from deepdiff import DeepHash

from nsight import exceptions, extraction, utils
from nsight.cache import GlobalNCUProfileCache
from nsight.collection import core
from nsight.exceptions import NCUErrorContext


def get_signature(
    func: Callable[..., Any],
    configs: List[Sequence[Any]],
) -> str:
    key = (func.__name__, configs)
    hashes = DeepHash(key)
    return cast(str, hashes[key])


def launch_ncu(
    report_path: str,
    func: Callable[..., None],
    configs: List[Sequence[Any]],
    metrics: Sequence[str],
    cache_control: Literal["none", "all"],
    clock_control: Literal["none", "base"],
    replay_mode: Literal["kernel", "range"],
    verbose: bool,
) -> str | None:
    """
    Launch NVIDIA Nsight Compute to profile the current script with specified options.

    Args:
        report_path: Path to write report file to.
        metrics: Specific metrics to collect.
        cache_control: Select cache control option
        clock_control: Select clock control option
        replay_mode: Select replay mode option
        verbose: If False, log is written to a file (ncu_log.txt)

    Raises:
        NCUNotAvailableError: If NCU is not available on the system.
        ProfilerException: If profiling fails due to an error from NVIDIA Nsight Compute.
        ValueError: If invalid values are provided for cache_control, clock_control, or replay_mode.

    Returns:
        path to the NVIDIA Nsight Compute log file
        Produces NVIDIA Nsight Compute report file with profiling data.
    """
    assert report_path.endswith(".ncu-rep")

    # Set an environment variable to track the main process ID.
    os.environ["NSPY_NCU_MAIN_PID"] = str(os.getpid())

    # Determine the script being executed
    script_path = os.path.abspath(sys.argv[0])
    script_args = " ".join(sys.argv[1:])

    # Set an environment variable to detect call site
    env = os.environ.copy()
    env["NSPY_NCU_PROFILE"] = get_signature(func, configs)

    if cache_control not in ("none", "all"):
        raise ValueError("cache_control must be 'none', or 'all'")
    if clock_control not in ("none", "base"):
        raise ValueError("clock_control must be 'none', or 'base'")
    if replay_mode not in ("kernel", "range"):
        raise ValueError("replay_mode must be 'kernel', or 'range'")

    cache = f"--cache-control {cache_control}"
    clocks = f"--clock-control {clock_control}"
    replay = f"--replay-mode {replay_mode}"
    log_path = os.path.splitext(report_path)[0] + ".log"
    log = f"--log-file {log_path}"
    nvtx = f'--nvtx --nvtx-include "regex:{utils.NVTX_DOMAIN}@.+/"'
    metrics_str = ",".join(metrics)

    # Construct the ncu command
    ncu_command = f"""ncu {log} {cache} {clocks} {replay} {nvtx} --metrics {metrics_str} -f -o {report_path} {sys.executable} {script_path} {script_args}"""

    # Check if ncu is available on the system
    ncu_available = False
    try:
        subprocess.run(
            ["ncu", "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        ncu_available = True
    except:
        ncu_available = False

    if ncu_available:
        try:
            subprocess.run(
                ncu_command,
                shell=True,
                check=True,
                env=env,
            )

            return log_path
        except subprocess.CalledProcessError as e:
            log_parser = utils.NCULogParser()
            error_logs = log_parser.get_logs(log_path, "ERROR")

            # Create error context
            error_context = NCUErrorContext(
                errors=error_logs,
                log_file_path=log_path,
                metrics=metrics,
            )

            error_message = utils.format_ncu_error_message(error_context)
            raise exceptions.ProfilerException(error_message) from None
    else:
        subprocess.run([sys.executable, script_path], env=env)
        raise exceptions.NCUNotAvailableError(
            "Nsight Compute CLI (ncu) is not available on this system. Profiling will not be performed.\n"
            "Please install Nsight Compute CLI."
        )


class NCUCollector(core.NsightCollector):
    """
    NCU collector for Nsight Python.

    Args:
        metrics: Metrics to collect from
            NVIDIA Nsight Compute. By default we collect kernel runtimes in nanoseconds.
            A list of supported metrics can be found with ``ncu --list-metrics``.
        ignore_kernel_list: List of kernel names to ignore.
            If you call a library within a ``annotation`` context, you might not have
            precise control over which and how many kernels are being launched.
            If some of these kernels should be ignored in the Nsight Python profile, their
            their names can be blacklisted. Default: ``None``
        combine_kernel_metrics: By default, Nsight Python
            expects one kernel launch per annotation. In case an annotated region launches
            multiple kernels, instead of failing the profiling run, you can specify
            how to summarize the collected metrics into a single number. For example,
            if we profile runtime and want to sum the times of all kernels we can specify
            ``combine_kernel_metrics = lambda x, y: x + y``. The function should take
            two arguments and return a single value. Default: ``None``.
        clock_control: Select clock_control option
            control in NVIDIA Nsight Compute. If ``None``, we launch ``ncu --clock-control none ...``.
            For more details, see the NVIDIA Nsight Compute Profiling Guide:
            https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#clock-control
            Default: ``None``
        cache_control: Select cache_control option
            control in NVIDIA Nsight Compute. If ``None``, we launch ``ncu --cache-control none ...``.
            For more details, see the NVIDIA Nsight Compute Profiling Guide:
            https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#cache-control
            Default: ``all``
        replay_mode: Select replay mode option
            control in NVIDIA Nsight Compute. If ``None``, we launch ``ncu --replay-mode kernel ...``.
            For more details, see the NVIDIA Nsight Compute Profiling Guide:
            https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#replay
            Default: ``kernel``
    """

    def __init__(
        self,
        metrics: Sequence[str] = ["gpu__time_duration.sum"],
        ignore_kernel_list: Sequence[str] | None = None,
        combine_kernel_metrics: Callable[[float, float], float] | None = None,
        clock_control: Literal["base", "none"] = "none",
        cache_control: Literal["all", "none"] = "all",
        replay_mode: Literal["kernel", "range"] = "kernel",
    ):
        if clock_control not in ("none", "base"):
            raise ValueError("clock_control must be 'none', or 'base'")
        if cache_control not in ("none", "all"):
            raise ValueError("cache_control must be 'none', or 'all'")
        if replay_mode not in ("kernel", "range"):
            raise ValueError("replay_mode must be 'kernel', or 'range'")

        self.metrics = metrics
        self.ignore_kernel_list = ignore_kernel_list or []
        self.combine_kernel_metrics = combine_kernel_metrics
        self.clock_control = clock_control
        self.cache_control = cache_control
        self.replay_mode = replay_mode

    def collect(
        self,
        func: Callable[..., None],
        configs: Iterable[Sequence[Any]],
        settings: core.ProfileSettings,
    ) -> pd.DataFrame | None:
        """
        Collects profiling data using NVIDIA Nsight Compute.

        Args:
            func: The function to profile.
            configs: iterable of configurations to run the function with.
            settings: Profiling settings.

        Returns:
            Collected profiling data.
        """

        # Materialize the configs
        configs_list = list(configs)

        # Check if the script is already running under ncu
        if "NSPY_NCU_PROFILE" not in os.environ:

            tag = f"{func.__name__}-{func._nspy_ncu_run_id}"  # type: ignore[attr-defined]
            report_path = f"{settings.output_prefix}ncu-output-{tag}.ncu-rep"

            # Launch NVIDIA Nsight Compute
            log_path = launch_ncu(
                report_path,
                func,
                configs_list,
                self.metrics,
                self.cache_control,
                self.clock_control,
                self.replay_mode,
                settings.output_detailed,
            )

            if settings.output_progress:
                print("[NSIGHT-PYTHON] Profiling completed successfully !")
                print(
                    f"[NSIGHT-PYTHON] Refer to {report_path} for the NVIDIA Nsight Compute CLI report"
                )
                print(
                    f"[NSIGHT-PYTHON] Refer to {log_path} for the NVIDIA Nsight Compute CLI logs"
                )

            df = extraction.extract_df_from_report(
                report_path,
                self.metrics,
                configs_list,  # type: ignore[arg-type]
                settings.runs,
                func,
                settings.derive_metric,
                self.ignore_kernel_list,  # type: ignore[arg-type]
                settings.output_progress,
                self.combine_kernel_metrics,
            )

            # Save to cache
            GlobalNCUProfileCache().save_profile_result(func.__name__, df)

            return df

        else:
            # If NSPY_NCU_PROFILE is set, just run the function normally
            sig = os.environ["NSPY_NCU_PROFILE"]

            # If this is not the function we are profiling, just load from cache
            if get_signature(func, configs_list) != sig:
                return GlobalNCUProfileCache().load_profile_result(func.__name__)

            if settings.output_progress:
                utils.print_header(
                    f"Profiling {func.__name__}",
                    f"{len(configs_list)} configurations, {settings.runs} runs each",
                )

            core.run_profile_session(
                func,
                configs_list,
                settings.runs,
                settings.output_progress,
                settings.output_detailed,
                settings.thermal_control,
            )

            # Exit after profiling to prevent the rest of the script from running
            # Use os._exit() instead of sys.exit() to avoid pytest catching SystemExit
            os._exit(0)
