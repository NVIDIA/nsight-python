# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

"""
Exceptions specific to Nsight Python profiling and analysis.
"""


class MetricErrorType(Enum):
    """
    Enum for different types of metric errors.
    """

    INVALID = "Invalid"
    UNSUPPORTED = "Unsupported"


class ProfilerException(Exception):
    """
    Exception raised for errors specific to the Profiler.

    Attributes:
        message: Explanation of the error.
    """

    pass


class NCUNotAvailableError(Exception):
    """
    Exception raised when NVIDIA Nsight Compute CLI (NCU) is not available or accessible.

    This can occur when:
    - NCU is not installed on the system
    - NCU is not in the system PATH
    - Required permissions are missing
    """

    pass


CUDA_CORE_UNAVAILABLE_MSG = "cuda-core is required for ignore_failures functionality.\n Install it with:\n  - pip install nsight-python[cu12]  (if you have CUDA 12.x)\n  - pip install nsight-python[cu13]  (if you have CUDA 13.x)"


def get_metrics_error_message(
    metrics: Sequence[str], error_type: MetricErrorType
) -> str:
    """
    Returns a formatted error message for invalid or unsupported metric names.

    Args:
        metrics: The invalid/unsupported metric names that was provided.
        error_type: The type of error (Invalid or Unsupported).

    Returns:
        str: User-friendly error message with guidance.
    """
    return (
        f"{error_type.value} value {metrics} for 'metrics' parameter for nsight.analyze.kernel()."
        f"\nPlease refer ncu --query-metrics for list of supported metrics."
    )


@dataclass
class NCUErrorContext:
    """
    Context information for NCU error handling.

    Attributes:
        errors: The error logs from NCU
        log_file_path: Path to the NCU log file
        metrics: The metrics that was being collected
    """

    errors: list[str]
    log_file_path: str
    metrics: Sequence[str]
