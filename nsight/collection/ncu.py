# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Collection utilities for profiling Nsight Python runs using NVIDIA Nsight Compute (ncu).

This module contains logic for launching NVIDIA Nsight Compute with appropriate settings.
NCU is instructed to profile specific code sections marked by NVTX ranges - the
Nsight Python annotations.
"""

import ctypes
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Collection, Iterable, Sequence
from enum import IntEnum
from typing import Any, Literal

import pandas as pd

from nsight import exceptions, extraction, utils
from nsight.collection import core
from nsight.exceptions import NCUErrorContext
from nsight.utils import VerbosityLevel

BEGIN_SYMBOL = "nvInjBeginProfiling"
STOP_SYMBOL = "nvInjEndProfiling"
MIN_NCU_VERSION = (2026, 2, 1, 0)

injection_load_error: (
    exceptions.ProfilerException | exceptions.NCUNotAvailableError | None
) = None

begin_profiling_symbol: Callable[..., Any] | None = None
end_profiling_symbol: Callable[..., Any] | None = None


# Parameter structs matching nvComputeInj.h
class ProfilingParams(ctypes.Structure):
    _fields_ = [
        ("structSize", ctypes.c_size_t),
        ("pPriv", ctypes.c_void_p),
    ]


PROFILING_PARAMS = ProfilingParams(
    structSize=ctypes.sizeof(ProfilingParams),
    pPriv=None,
)


class NvInjResult(IntEnum):
    NV_INJ_SUCCESS = 0
    NV_INJ_ERROR_UNKNOWN = 1
    NV_INJ_ERROR_INVALID_PARAMS = 2
    NV_INJ_ERROR_NOT_INITIALIZED = 3


def check_ncu_version(ncu_path: str) -> None:
    """Raise ProfilerException if the installed ncu is older than MIN_NCU_VERSION."""
    try:
        result = subprocess.run(
            [ncu_path, "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return

    # Matches e.g. "Version 2026.2.1.0" and captures the four numeric components
    match = re.search(r"Version\s+(\d+)\.(\d+)\.(\d+)\.(\d+)", result.stdout)
    if match:
        version = tuple(int(x) for x in match.groups())
        if version < MIN_NCU_VERSION:
            min_str = ".".join(str(x) for x in MIN_NCU_VERSION)
            got_str = ".".join(str(x) for x in version)
            raise exceptions.ProfilerException(
                f"NVIDIA Nsight Compute {min_str} or later is required "
                f"(CUDA Toolkit 13.3 Update 1 or later), but found {got_str}. "
                f"Please upgrade NVIDIA Nsight Compute."
            )


def format_injection_error_message(code: int, action: str) -> str:
    try:
        name = NvInjResult(code).name
    except ValueError:
        name = "<unknown>"

    return f"Failed to {action} profiling via Nsight Compute injection: {name} (code={code})."


def get_injection_library_path(ncu_path: str) -> str:
    result = subprocess.run(
        [ncu_path, "--list-injection-path-64"],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    lines = result.stdout.splitlines()
    if not lines:
        raise exceptions.ProfilerException("Failed to query the NCU injection library")
    path = lines[0].strip()
    if not path or not os.path.isdir(path):
        raise exceptions.ProfilerException(
            f"Failed to find the NCU injection library at {path}"
        )
    return path


def load_library(path: str) -> ctypes.CDLL:
    """Load the injection shared library."""
    if os.name != "posix":
        raise exceptions.ProfilerException(f"Unsupported operating system: {os.name}")
    try:
        return ctypes.CDLL(path)
    except OSError as exc:
        raise exceptions.ProfilerException(
            f"Failed to load the NCU injection library at {path}"
        ) from exc


def try_init_injection() -> None:
    """Load the injection library callables, or set ``injection_load_error`` on failure."""
    global injection_load_error
    global begin_profiling_symbol
    global end_profiling_symbol

    ncu_path = shutil.which("ncu")
    if ncu_path is None:
        injection_load_error = exceptions.NCUNotAvailableError(
            "Nsight Compute CLI (ncu) is not available on this system. Profiling will not be performed.\n"
            "Please install Nsight Compute CLI."
        )
        return

    try:
        check_ncu_version(ncu_path)

        inj_dir = get_injection_library_path(ncu_path)
        inj_lib_path = os.path.join(inj_dir, "libcuda-injection.so")
        if not os.path.isfile(inj_lib_path):
            raise exceptions.ProfilerException("Failed to find NCU injection library")

        os.environ["NVTX_INJECTION64_PATH"] = inj_lib_path
        os.environ["NV_COMPUTE_PROFILER_PERFWORKS_DIR"] = inj_dir
        os.environ["NV_COMPUTE_PROFILER_IN_PROCESS"] = "1"

        lib = load_library(inj_lib_path)

        begin_profiling_symbol = getattr(lib, BEGIN_SYMBOL, None)
        end_profiling_symbol = getattr(lib, STOP_SYMBOL, None)
        if begin_profiling_symbol is None or end_profiling_symbol is None:
            raise exceptions.ProfilerException(
                "Failed to find symbols in the NCU injection library: "
                f"need {BEGIN_SYMBOL!r} and {STOP_SYMBOL!r}"
            )

    except exceptions.ProfilerException as exc:
        injection_load_error = exc
        return

    except Exception as exc:
        injection_load_error = exceptions.ProfilerException(
            "Failed to load NCU injection library"
        )
        injection_load_error.__cause__ = exc
        injection_load_error.__suppress_context__ = True
        return

    injection_load_error = None


def launch_ncu(
    report_path: str,
    metrics: Sequence[str],
    cache_control: Literal["none", "all"],
    clock_control: Literal["none", "base"],
    replay_mode: Literal["kernel", "range"],
    verbosity: VerbosityLevel,
) -> tuple[subprocess.Popen[Any], str, str]:
    """
    Launch NVIDIA Nsight Compute to profile the current script with specified options.

    Args:
        report_path: Path to write report file to.
        metrics: Specific metrics to collect.
        cache_control: Select cache control option
        clock_control: Select clock control option
        replay_mode: Select replay mode option
        verbosity: Controls output verbosity. ``SILENT`` disables NCU logs
            (``--quiet``), which reduces detail in ``ProfilerException`` on failure.
            ``DEBUG`` prints the NCU command and enables ``--verbose`` NCU output.

    Raises:
        ValueError: If invalid values are provided for cache_control, clock_control, or replay_mode.

    Note:
        The attach command uses the ``ncu`` name on ``$PATH`` (no resolved full path).
        Whether NCU is usable is determined in :func:`try_init_injection` at import;
        ``NCUCollector.collect`` raises if the CLI was not found.

    Returns:
        path to the NVIDIA Nsight Compute log file
        Produces NVIDIA Nsight Compute report file with profiling data.
    """
    assert report_path.endswith(".ncu-rep")

    if cache_control not in ("none", "all"):
        raise ValueError("cache_control must be 'none', or 'all'")
    if clock_control not in ("none", "base"):
        raise ValueError("clock_control must be 'none', or 'base'")
    if replay_mode not in ("kernel", "range"):
        raise ValueError("replay_mode must be 'kernel', or 'range'")

    log_path = os.path.splitext(report_path)[0] + ".log"

    # Ensures ncu attaches to the correct process when multiple attachable processes exist
    target_pid = os.getpid()

    ncu_cmd = [
        "ncu",
        "--mode",
        "attach",
        "--process-id",
        str(target_pid),
        "--nvtx-include",
        f"regex:{utils.NVTX_DOMAIN}@.+/",
        "--log-file",
        log_path,
        "--cache-control",
        cache_control,
        "--clock-control",
        clock_control,
        "--replay-mode",
        replay_mode,
        *(["--verbose"] if verbosity >= VerbosityLevel.DEBUG else []),
        "--metrics",
        ",".join(metrics),
        "-f",
        "-o",
        report_path,
    ]

    if verbosity >= VerbosityLevel.DEBUG:
        print(f"[NSIGHT-PYTHON] NCU command: {' '.join(ncu_cmd)}")

    # Start ncu in the background so this process can run the profile session
    ncu_process = subprocess.Popen(ncu_cmd)
    return ncu_process, report_path, log_path


def begin_profiling() -> None:
    assert begin_profiling_symbol is not None
    rc = int(begin_profiling_symbol(ctypes.byref(PROFILING_PARAMS)))
    if rc != int(NvInjResult.NV_INJ_SUCCESS):
        raise exceptions.ProfilerException(format_injection_error_message(rc, "begin"))


def end_profiling() -> None:
    assert end_profiling_symbol is not None
    rc = int(end_profiling_symbol(ctypes.byref(PROFILING_PARAMS)))
    if rc != int(NvInjResult.NV_INJ_SUCCESS):
        raise exceptions.ProfilerException(format_injection_error_message(rc, "end"))


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

        Raises:
            exceptions.NCUNotAvailableError:
                Nsight Compute was not found on ``$PATH`` during import-time initialization
                (see :func:`try_init_injection`).
            exceptions.ProfilerException:
                Injection could not be loaded or NVTX injection failed around the
                profiled region; the ``ncu`` attach process exited with an error; the
                profiled function or ``configs`` are invalid (see
                :func:`nsight.collection.core.run_profile_session`); or the report
                file was missing or could not be parsed (see
                :func:`nsight.extraction.extract_df_from_report`).
            RuntimeError:
                Report extraction found no kernels, a mismatch in kernel counts, or
                several kernels in one annotation without
                ``combine_kernel_metrics`` (see :func:`nsight.extraction.extract_df_from_report`).
            TypeError, ValueError:
                From report extraction if ``settings.derive_metric`` is not a valid
                callable or its signature does not match the metrics and config tuple.
            exceptions.CoolingTimeoutError:
                If thermal throttling is enabled and the GPU does not reach the
                target headroom within ``thermal_timeout`` (see
                :class:`nsight.thermovision.ThermalController`).

        Note:
            Exceptions raised by ``func`` or by code it calls propagate to the
            caller unchanged.
        """

        if injection_load_error is not None:
            raise injection_load_error

        # Materialize the configs
        configs_list = list(configs)

        tag = f"{func.__name__}-{func._nspy_ncu_run_id}"  # type: ignore[attr-defined]
        report_path = f"{settings.output_prefix}ncu-output-{tag}.ncu-rep"

        begin_profiling()

        try:
            # Launch NVIDIA Nsight Compute
            ncu_process, report_path, log_path = launch_ncu(
                report_path,
                self.metrics,
                self.cache_control,
                self.clock_control,
                self.replay_mode,
                settings.verbosity,
            )

            if settings.verbosity >= VerbosityLevel.INFO:
                utils.print_header(
                    f"Profiling {func.__name__}",
                    f"{len(configs_list)} configurations, {settings.runs} runs each",
                )

            core.run_profile_session(
                func,
                configs_list,
                settings.runs,
                settings.verbosity,
                settings.thermal_mode,
                settings.thermal_wait,
                settings.thermal_cont,
                settings.thermal_timeout,
            )
        finally:
            end_profiling()

        return_code = ncu_process.wait()
        if return_code != 0:
            log_parser = utils.NCULogParser()
            error_logs = log_parser.get_logs(log_path, "ERROR")
            error_context = NCUErrorContext(
                errors=error_logs,
                log_file_path=log_path,
                metrics=self.metrics,
            )
            error_message = utils.format_ncu_error_message(error_context)
            raise exceptions.ProfilerException(error_message)

        if settings.verbosity >= VerbosityLevel.INFO:
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
            settings.verbosity,
            self.combine_kernel_metrics,
        )

        return df
