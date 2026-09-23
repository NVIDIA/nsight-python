# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Collection utilities for profiling Nsight Python runs using NVIDIA CUPTI.

This module contains logic for launching NVIDIA CUPTI with appropriate settings.
CUPTI is instructed to profile specific code sections marked by NVTX ranges - the
Nsight Python annotations.
"""

import bisect
import ctypes
import os
from collections.abc import Callable, Iterable, Sequence
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from cuda.bindings.runtime import (
    cudaDeviceSynchronize,
    cudaError_t,
    cudaGetErrorString,
)
from cuda.pathfinder import load_nvidia_dynamic_lib

from nsight import exceptions, extraction, utils
from nsight.collection import core
from nsight.exceptions import CUPTI_UNAVAILABLE_MSG
from nsight.utils import VerbosityLevel

try:
    from cupti import cupti

    CUPTI_AVAILABLE = True
except ImportError:
    CUPTI_AVAILABLE = False
    cupti = None

# Module-level state set up by init() and torn down by finalize()
_subscriber: int | None = None
# What NVTX_INJECTION64_PATH held before init() pointed it at CUPTI.
_previous_injection_path: str | None = None


class _MarkerRecord(NamedTuple):
    """One end of an NVTX range, already filtered to the Nsight Python domain."""

    marker_id: int
    timestamp: int
    is_start: bool
    name: str


class _LaunchRecord(NamedTuple):
    """A kernel-launching driver call, timestamped on the CPU."""

    correlation_id: int
    start: int


class _KernelRecord(NamedTuple):
    """A kernel that ran on the GPU, linked to its launch by correlation id."""

    correlation_id: int
    name: str
    duration: int


class _Annotation(NamedTuple):
    """An annotation's CPU-side open/close interval."""

    name: str
    start: int
    end: int


def _build_annotation_intervals(markers: list[_MarkerRecord]) -> list[_Annotation]:
    """Pair up start and end markers into annotation intervals.

    Args:
        markers: Range start and end markers, in any order.

    Returns:
        One interval per range that both opened and closed, carrying the
        annotation name and its CPU start and end timestamps. Unpaired or
        unnamed ranges are dropped, so the result may be shorter than the
        number of start markers.
    """
    starts: dict[int, _MarkerRecord] = {}
    ends: dict[int, int] = {}
    for marker in markers:
        if marker.is_start:
            starts[marker.marker_id] = marker
        else:
            ends[marker.marker_id] = marker.timestamp

    intervals = []
    for marker_id, start in starts.items():
        end = ends.get(marker_id)
        # An unclosed range is dropped rather than left open to the end of the
        # session: extending it would swallow every later launch and quietly
        # mis-attribute them, whereas dropping it surfaces as a kernel-count
        # error in extraction.
        if end is None or not start.name:
            continue
        intervals.append(_Annotation(start.name, start.timestamp, end))
    return intervals


def _attribute_kernels(
    markers: list[_MarkerRecord],
    launches: list[_LaunchRecord],
    kernels: list[_KernelRecord],
) -> list[tuple[str, str, int]]:
    """Work out which annotation each kernel belongs to.

    A kernel belongs to the annotation that was open when it was *launched*,
    not when it ran: launches are asynchronous, so a kernel can execute well
    after its range has closed. Launch timestamps and marker timestamps are
    both CPU-side, so they can be compared directly, and the correlation id
    then links the launch to the kernel that came out of it.

    Args:
        markers: Range start and end markers, already filtered to the
            Nsight Python NVTX domain.
        launches: Kernel-launching driver calls, with CPU timestamps.
        kernels: Kernels that ran on the GPU.

    Returns:
        One ``(annotation, kernel_name, duration_ns)`` tuple per attributed
        kernel, ready to hand to ``add_profiling_data``. Ordered by launch
        time, which is the order the profiling loop ran in and therefore the
        order extraction expects: it reads this list positionally to decide
        which (config, run) each measurement belongs to.

        Kernels that cannot be attributed are left out entirely: a launch
        made outside any annotation, or one whose range never closed.
    """
    # Annotations can neither nest nor overlap, so these intervals are disjoint
    # and can be searched by start time rather than scanned: the only candidate
    # for a launch is the last interval that opened at or before it.
    intervals = sorted(_build_annotation_intervals(markers), key=lambda a: a.start)
    interval_starts = [interval.start for interval in intervals]

    # Indexed so each launch can find its kernel without rescanning the list.
    kernels_by_correlation = {kernel.correlation_id: kernel for kernel in kernels}

    attributed: list[tuple[str, str, int]] = []
    for launch in sorted(launches, key=lambda launch: launch.start):
        # Which annotation was open at this instant.
        index = bisect.bisect_right(interval_starts, launch.start) - 1
        if index < 0:
            continue
        annotation = intervals[index]
        if launch.start > annotation.end:
            continue
        kernel = kernels_by_correlation.get(launch.correlation_id)
        if kernel is not None:
            attributed.append((annotation.name, kernel.name, kernel.duration))
    return attributed


def _restore_injection_path(previous: str | None) -> None:
    """Put NVTX_INJECTION64_PATH back to ``previous``, unsetting it if unset."""
    if previous is None:
        os.environ.pop("NVTX_INJECTION64_PATH", None)
    else:
        os.environ["NVTX_INJECTION64_PATH"] = previous


def init() -> None:
    global _subscriber, _previous_injection_path

    if not CUPTI_AVAILABLE:
        raise ImportError(CUPTI_UNAVAILABLE_MSG)

    # cupti-python should ideally expose an API for this instead of
    # requiring us to resolve the path manually via cuda-pathfinder.
    abs_path = load_nvidia_dynamic_lib("cupti").abs_path
    if abs_path is None:
        raise RuntimeError(
            "cuda.pathfinder could not resolve an absolute path for cupti"
        )
    _previous_injection_path = os.environ.get("NVTX_INJECTION64_PATH")
    os.environ["NVTX_INJECTION64_PATH"] = abs_path

    params = cupti.SubscriberParams()
    params.struct_size = cupti.SUBSCRIBER_PARAMS_SIZE
    params.subscriber_name = "nsight-python-cupti"
    try:
        _subscriber = cupti.subscribe_v2(None, None, params.ptr)
    except Exception:
        # The injection path is only ours to set while we hold the CUPTI
        # subscription.
        _restore_injection_path(_previous_injection_path)
        raise


def finalize() -> None:
    global _subscriber, _previous_injection_path

    if not CUPTI_AVAILABLE:
        raise ImportError(CUPTI_UNAVAILABLE_MSG)

    cupti.finalize()
    _subscriber = None
    _restore_injection_path(_previous_injection_path)
    _previous_injection_path = None


class CUPTICollector(core.NsightCollector):
    """
    Experimental CUPTI-based collector for Nsight Python.

    Uses CUPTI activity records to measure kernel durations directly, without
    launching ``ncu``. This makes it lower-overhead than the NCU collector but
    comes with the following limitations:

    **Supported metrics:** Only ``"gpu__time_duration.sum"`` is supported. CUPTI
    measures this by recording kernel start and end timestamps from CUPTI activity
    records (``CONCURRENT_KERNEL``), computing ``end - start`` per kernel launch.

    **Replay mode:** Neither kernel replay nor range replay is supported. CUPTI
    activity records capture one execution of each kernel in a single pass; no
    replay mechanism is engaged.

    **Clock and cache control:** ``clock_control`` and ``cache_control`` are not
    supported. Clocks are not locked and caches are not flushed between runs,
    which may affect measurement stability. Use :attr:`ProfileSettings.runs` with
    a sufficient number of repetitions and check ``StableMeasurement`` in the
    output to detect high variance.

    .. note::
        The whole device is synchronized once after the profiling session, before
        the activity buffers are flushed, because a kernel's activity record is
        only written when the kernel completes. Work the profiled function left
        running asynchronously is therefore waited for before profiling returns.

    Args:
        metrics: Must be ``["gpu__time_duration.sum"]``.
        ignore_kernel_list: List of kernel names to ignore.
            If you call a library within a ``annotation`` context, you might not have
            precise control over which and how many kernels are being launched.
            If some of these kernels should be ignored in the Nsight Python profile,
            their names can be provided here. Default: ``None``
        combine_kernel_metrics: By default, Nsight Python
            expects one kernel launch per annotation. In case an annotated region launches
            multiple kernels, instead of failing the profiling run, you can specify
            how to summarize the collected metrics into a single number. For example,
            if we profile runtime and want to sum the times of all kernels we can specify
            ``combine_kernel_metrics = lambda x, y: x + y``. The function should take
            two arguments and return a single value. Default: ``None``.
    """

    def __init__(
        self,
        metrics: Sequence[str] = ["gpu__time_duration.sum"],
        ignore_kernel_list: Sequence[str] | None = None,
        combine_kernel_metrics: Callable[[float, float], float] | None = None,
    ):
        if not CUPTI_AVAILABLE:
            raise ImportError(CUPTI_UNAVAILABLE_MSG)
        if len(metrics) > 1 or metrics[0] != "gpu__time_duration.sum":
            raise ValueError("metrics must be ['gpu__time_duration.sum']")
        self.metrics = metrics
        self.ignore_kernel_list = ignore_kernel_list or []
        self.combine_kernel_metrics = combine_kernel_metrics

    def collect(
        self,
        func: Callable[..., None],
        configs: Iterable[Sequence[Any]],
        settings: core.ProfileSettings,
    ) -> pd.DataFrame | None:
        """
        Collects profiling data using NVIDIA CUPTI.

        Args:
            func: The function to profile.
            configs: iterable of configurations to run the function with.
            settings: Profiling settings.

        Returns:
            Collected profiling data.
        """

        # Materialize the configs
        configs_list = list(configs)

        if settings.verbosity >= VerbosityLevel.INFO:
            utils.print_header(
                f"Profiling {func.__name__}",
                f"{len(configs_list)} configurations, {settings.runs} runs each",
            )

        profiling_data: dict[str, list[utils.ActionData]] = {}

        gpu, compute_clock, memory_clock = utils.get_device_properties()

        def add_profiling_data(annotation: str, kernel_name: str, value: int) -> None:
            if (
                kernel_name is None
                or len(kernel_name) == 0
                or kernel_name in self.ignore_kernel_list
            ):
                return
            if annotation not in profiling_data:
                profiling_data[annotation] = []
            failure = utils.DUMMY_KERNEL_NAME in kernel_name
            profiling_data[annotation].append(
                utils.ActionData(
                    name=kernel_name,
                    values=None if failure else np.array([value]),
                    compute_clock=compute_clock,
                    memory_clock=memory_clock,
                    gpu=gpu,
                    units=["ns"],
                )
            )

        def func_buffer_requested() -> tuple[int, int]:
            buffer_size = 8 * 1024 * 1024  # 8MB buffer
            max_num_records = 0
            return buffer_size, max_num_records

        marker_records: list[_MarkerRecord] = []
        launch_records: list[_LaunchRecord] = []
        kernel_records: list[_KernelRecord] = []

        def func_buffer_completed(activities: list[Any]) -> None:
            for activity in activities:
                kind = activity.kind
                if kind == cupti.ActivityKind.CONCURRENT_KERNEL:
                    kernel_records.append(
                        _KernelRecord(
                            activity.correlation_id,
                            activity.name,
                            activity.end - activity.start,
                        )
                    )
                elif kind == cupti.ActivityKind.DRIVER:
                    # Only the launch cbids are enabled, so no filtering needed.
                    launch_records.append(
                        _LaunchRecord(activity.correlation_id, activity.start)
                    )
                elif kind == cupti.ActivityKind.MARKER:
                    if activity.domain != utils.NVTX_DOMAIN:
                        continue
                    marker_records.append(
                        _MarkerRecord(
                            activity.id,
                            activity.timestamp,
                            bool(
                                int(activity.flags_)
                                & int(cupti.ActivityFlag.MARKER_START)
                            ),
                            activity.name,
                        )
                    )

        cupti.activity_register_callbacks(func_buffer_requested, func_buffer_completed)

        # The kernel-launch driver APIs whose activity records we need
        _LAUNCH_CBIDS = (
            cupti.Driver_api_trace_cbid.cuLaunchKernel,
            cupti.Driver_api_trace_cbid.cuLaunchKernel_ptsz,
            cupti.Driver_api_trace_cbid.cuLaunchKernelEx,
            cupti.Driver_api_trace_cbid.cuLaunchKernelEx_ptsz,
            cupti.Driver_api_trace_cbid.cuLaunchCooperativeKernel,
            cupti.Driver_api_trace_cbid.cuLaunchCooperativeKernel_ptsz,
        )

        # Everything enabled below is disabled again in the finally, so a
        # failure part way through enabling does not leave some of it on.
        try:
            cupti.activity_enable(cupti.ActivityKind.CONCURRENT_KERNEL)
            cupti.activity_enable(cupti.ActivityKind.MARKER)

            # Driver activity records carry the CPU timestamp of each launch,
            # which is what ties a kernel to the annotation it was launched
            # inside. Enable them per cbid; enabling the whole DRIVER kind
            # would trace every call.
            for cbid in _LAUNCH_CBIDS:
                cupti.activity_enable_driver_api(cbid, 1)

            core.run_profile_session(
                func,
                configs_list,
                settings.runs,
                settings.verbosity,
                settings.thermal_mode,
                settings.thermal_wait,
                settings.thermal_cont,
                settings.thermal_timeout,
                settings.thermal_device,
                settings.info_collectors,
                settings.output_prefix,
            )

            # A kernel's activity record only exists once it has finished, so
            # wait for the device before flushing.
            (sync_status,) = cudaDeviceSynchronize()
            if sync_status != cudaError_t.cudaSuccess:
                message = cudaGetErrorString(sync_status)[1].decode()
                raise exceptions.ProfilerException(
                    "Failed to synchronize the device before collecting CUPTI "
                    f"activity records: {message}"
                )

        finally:
            # Drain the buffers whatever happened: records left in them are
            # delivered to whichever session flushes next, and would be counted
            # as that session's kernels.
            cupti.activity_flush_all(1)
            # Disabling something that was never enabled is a no-op, so this is
            # safe however far the block above got.
            for cbid in _LAUNCH_CBIDS:
                cupti.activity_enable_driver_api(cbid, 0)
            cupti.activity_disable(cupti.ActivityKind.CONCURRENT_KERNEL)
            cupti.activity_disable(cupti.ActivityKind.MARKER)

        # CUPTI drops a record when it has nowhere to put it. Attributing an
        # incomplete set would silently under-report kernels, so stop instead.
        dropped = ctypes.c_size_t(0)
        cupti.activity_get_num_dropped_records(0, 0, ctypes.addressof(dropped))
        if dropped.value:
            raise exceptions.ProfilerException(
                f"The cupti tool dropped {dropped.value} activity records "
                "because CUPTI was not given buffer space to write them into, "
                "so the profile is incomplete."
            )

        for annotation, kernel_name, duration in _attribute_kernels(
            marker_records, launch_records, kernel_records
        ):
            add_profiling_data(annotation, kernel_name, duration)

        if settings.verbosity >= VerbosityLevel.INFO:
            print("[NSIGHT-PYTHON] Profiling completed successfully !")

        # Numeric values collected per configuration, run, or annotation vary
        # across raw rows and are aggregated alongside profiling metrics.
        config_scope_columns = []
        annotation_scope_columns = []
        if settings.info_collectors:
            for name, _callback, scope in settings.info_collectors:
                if scope in ("config", "run"):
                    config_scope_columns.append(name)
                elif scope == "annotation":
                    annotation_scope_columns.append(name)

        df = extraction.extract_df_from_data(
            profiling_data,
            self.metrics,
            configs_list,  # type: ignore[arg-type]
            settings.runs,
            func,
            settings.derive_metric,
            settings.verbosity,
            self.combine_kernel_metrics,
            settings.info_collectors,
            config_scope_columns,
            annotation_scope_columns,
            info_prefix=settings.output_prefix or "",
        )

        return df
