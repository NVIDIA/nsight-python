# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Access the log produced by the active Compute Sanitizer."""

from pathlib import Path

from nsight import exceptions
from nsight.collection import compute_sanitizer


def log_path() -> Path:
    """Return the text log destination selected during sanitizer activation.

    The frontend updates the log while the Python process runs.
    """
    from nsight.tools_manager import Tool, get_active_tool

    if get_active_tool() != Tool.COMPUTE_SANITIZER:
        raise exceptions.ProfilerException(
            "Compute Sanitizer is not active. Call "
            "nsight.activate(nsight.Tool.COMPUTE_SANITIZER) before CUDA use."
        )
    return compute_sanitizer.get_log_path()
