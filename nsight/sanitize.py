# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Access the log produced by the active Compute Sanitizer."""

from pathlib import Path

from nsight import exceptions
from nsight.collection import compute_sanitizer


def log_path() -> Path:
    """Return the path of the text log written by the active Compute Sanitizer.

    The path is the destination selected during activation: either the
    ``log_path`` passed in :class:`nsight.ComputeSanitizerOptions` or a unique
    file in the system temporary directory. The sanitizer frontend updates the
    file while the Python process runs.

    Returns:
        The log destination as a :class:`pathlib.Path`. The Compute Sanitizer
        frontend may not have created the file yet, so check
        :meth:`pathlib.Path.exists` before reading it.

    Raises:
        ProfilerException: If Compute Sanitizer is not active, is not fully
            initialized, or its frontend has already exited.
    """
    from nsight.tools_manager import Tool, get_active_tool

    if get_active_tool() != Tool.COMPUTE_SANITIZER:
        raise exceptions.ProfilerException(
            "Compute Sanitizer is not active. Call "
            "nsight.activate(nsight.Tool.COMPUTE_SANITIZER) before CUDA use."
        )
    return compute_sanitizer.get_log_path()
