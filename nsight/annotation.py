# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import threading
from collections.abc import Callable
from typing import Any

import nvtx

import nsight.utils as utils
from nsight import info_collector
from nsight.exceptions import CUDA_CORE_UNAVAILABLE_MSG

# Thread-local storage for tracking annotations
# Each thread maintains its own annotation state to prevent
# conflicts in multi-threaded profiling scenarios
_thread_local_storage = threading.local()


def _is_in_annotation() -> bool:
    """
    Check if we are currently inside an annotation context.

    Returns:
        True if currently inside an annotation, False otherwise.
    """
    if not hasattr(_thread_local_storage, "in_annotation"):
        _thread_local_storage.in_annotation = False
    return bool(_thread_local_storage.in_annotation)


def _set_in_annotation(value: bool) -> None:
    """
    Set whether we are currently inside an annotation context.

    Args:
        value: True if entering an annotation, False if exiting.
    """
    _thread_local_storage.in_annotation = value


def add_active_annotation(name: str) -> None:
    """
    Add an annotation name to the used annotations set for the current thread.

    This function is called when entering an annotation context to track which
    annotation names have been used in the current profiling run. This prevents
    duplicate annotation names from being used within the same profiling run.

    Args:
        name: The annotation name to mark as used.
    """
    if not hasattr(_thread_local_storage, "used_annotation_names"):
        _thread_local_storage.used_annotation_names = set()
    _thread_local_storage.used_annotation_names.add(name)


def clear_active_annotations() -> None:
    """
    Clear all annotation state for the current thread.

    This function is called by the profiler at the start of each profiling run
    to reset the annotation state. This allows annotation names to be reused
    across different profiling runs.

    Note:
        This is automatically called by ``nsight.analyze.kernel`` before each run,
        so users typically don't need to call this manually.
    """
    if hasattr(_thread_local_storage, "used_annotation_names"):
        _thread_local_storage.used_annotation_names.clear()
    _set_in_annotation(False)


def is_active_annotation(name: str) -> bool:
    """
    Check if an annotation name has been used in the current profiling run.

    Args:
        name: The annotation name to check.

    Returns:
        True if the annotation name has been used, False otherwise.
    """
    if hasattr(_thread_local_storage, "used_annotation_names"):
        return name in _thread_local_storage.used_annotation_names

    return False


class annotate:
    """
    A decorator/context-manager hybrid for marking profiling regions.
    The encapsulated code will be profiled and associated with an NVTX
    range of the given annotate name.

    Example usage::

        # as context manager
        with nsight.annotate("name"):
            # your kernel launch here

        # as decorator
        @nsight.annotate("name")
        def your_kernel_launcher(...):
            ...

    Args:
        name: Name of the annotation to be used for profiling. Annotation names
            must be unique within a single profiling run to ensure unambiguous results.
        ignore_failures: Flag indicating whether to ignore
            failures in the annotate context. If set to ``True``, any exceptions
            raised within the context will be ignored, and the profiling will
            continue. The measured metric for this run will be set to NaN.
            Default: ``False``

    Raises:
        ValueError: If attempting to enter a nested annotation context, or if
            an annotation with the same name has already been used in this
            profiling run. Nested annotations are not supported and annotation
            names must be unique within a profiling run to avoid incorrect
            profiling results.
    Note:
        All annotations are created under the NVTX domain ``"nsight-python"``.
        This domain is used internally to filter and identify Nsight Python
        annotations in profiling tools.

    Note:
        Nested annotations are currently not supported. However, since each
        annotation is expected to contain a single kernel launch by default,
        nested annotations should not be necessary in typical usage scenarios.
        Entering and exiting the annotation context starts and ends an NVTX
        range that is visible across all threads in the profiled process, so
        kernels launched from worker threads within the annotated region are
        also captured.

    """

    def __init__(self, name: str, ignore_failures: bool = False):
        self.name = name
        self.ignore_failures = ignore_failures

        # Check if cuda-core is available when ignore_failures is True
        if ignore_failures and not utils.CUDA_CORE_AVAILABLE:
            raise ImportError(CUDA_CORE_UNAVAILABLE_MSG)

    def __enter__(self) -> "annotate":

        # Check for nested annotations
        if _is_in_annotation():
            raise ValueError(
                f"Nested annotations are not supported. "
                f"Cannot enter annotation '{self.name}' while another annotation is active."
            )

        # Check for duplicate annotation names
        if is_active_annotation(self.name):
            raise ValueError(
                f"Annotation name '{self.name}' has already been used in this profiling run. "
                f"Each annotation must have a unique name."
            )

        add_active_annotation(self.name)
        _set_in_annotation(True)
        # Collect annotation-scope information
        info_collector.collect_for_annotation(self.name)
        self._range_id = nvtx.start_range(message=self.name, domain=utils.NVTX_DOMAIN)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        try:
            if exc_type and self.ignore_failures:
                utils.launch_dummy_kernel_module()
        finally:
            nvtx.end_range(self._range_id)
            _set_in_annotation(False)

        if exc_type and not self.ignore_failures:
            return False  # propagate the exception

        return True

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with self:
                return func(*args, **kwargs)

        return wrapped
