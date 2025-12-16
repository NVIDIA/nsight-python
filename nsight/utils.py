# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import inspect
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import islice
from typing import Any, Iterator

import numpy as np
from numpy.typing import NDArray

from nsight.exceptions import (
    CUDA_CORE_UNAVAILABLE_MSG,
    MetricErrorType,
    NCUErrorContext,
    get_metrics_error_message,
)

# Try to import cuda-core (optional dependency)
try:
    from cuda.core.experimental import (
        Device,
        LaunchConfig,
        Program,
        ProgramOptions,
        launch,
    )

    CUDA_CORE_AVAILABLE = True
except ImportError:
    CUDA_CORE_AVAILABLE = False
    Device = None
    LaunchConfig = None
    Program = None
    ProgramOptions = None
    launch = None

NVTX_DOMAIN = "nsight-python"


class row_panel:
    pass


class col_panel:
    pass


class Colors:
    """For colorful printing."""

    HEADER = "\033[95m"
    BLUE = "\033[0;34m"
    CYAN = "\033[0;36m"
    GREEN = "\033[0;32m"
    ORANGE = "\033[0;33m"
    RED = "\033[0;31m"
    PURPLE = "\033[0;35m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def purple(msg: str) -> str:  # pragma: no cover
    """Prints ``msg`` in purple."""
    return Colors.PURPLE + msg + Colors.ENDC


# ------------------------------------------------------------------------------


@functools.lru_cache
def get_dummy_kernel_module() -> Any:
    """
    Returns a dummy kernel that does nothing.  In case a provider fails for some, reason, but we
    want to keep benchmarking we launch this dummy kernel such that during our later analysis of the
    ncu-report we still find the expected number of measured kernels per provider.

    The measured runtime of this kernel is ignored and the final result of the failed run will be
    reported as NaN.

    Raises:
        ImportError: If cuda-core is not installed.
    """
    if not CUDA_CORE_AVAILABLE:
        raise ImportError(CUDA_CORE_UNAVAILABLE_MSG)
    code = "__global__ void dummy_kernel_failure() {}"
    program_options = ProgramOptions(std="c++17")
    prog = Program(code, code_type="c++", options=program_options)
    return prog.compile("cubin", name_expressions=("dummy_kernel_failure",))


def launch_dummy_kernel_module() -> None:
    """
    Launch a dummy kernel module.

    Raises:
        ImportError: If cuda-core is not installed.
    """
    if not CUDA_CORE_AVAILABLE:
        raise ImportError(CUDA_CORE_UNAVAILABLE_MSG)
    dev = Device()
    dev.set_current()
    stream = dev.create_stream()
    mod = get_dummy_kernel_module()
    kernel = mod.get_kernel("dummy_kernel_failure")
    config = LaunchConfig(grid=1, block=256)
    launch(stream, config, kernel)
    stream.sync()


def format_time(seconds: float) -> str:
    """Convert ``seconds`` into ``HH:MM:SS`` format"""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


# Sincerely stolen (and adjusted) from attention-gym
def print_header(*lines: str) -> None:
    width = max(80, max(len(line) for line in lines) + 4)
    print(purple("╔" + "═" * (width - 2) + "╗"))
    for line in lines:
        print(purple(f"║ {line.center(width - 4)} ║"))
    print(purple("╚" + "═" * (width - 2) + "╝"))


@dataclass
class NCUActionData:
    name: str
    values: NDArray[Any] | None
    compute_clock: int
    memory_clock: int
    gpu: str

    @staticmethod
    def combine(value_reduce_op: Any) -> Any:
        """
        Combines two NCUActionData objects into a new one by applying the
        value_reduce_op to their values.
        """

        def _combine(lhs: "NCUActionData", rhs: "NCUActionData") -> "NCUActionData":
            assert lhs.compute_clock == rhs.compute_clock
            assert lhs.memory_clock == rhs.memory_clock
            assert lhs.gpu == rhs.gpu
            return NCUActionData(
                name=f"{lhs.name}|{rhs.name}",
                values=value_reduce_op(lhs.values, rhs.values),
                compute_clock=lhs.compute_clock,
                memory_clock=lhs.memory_clock,
                gpu=lhs.gpu,
            )

        return _combine


def print_progress_bar(
    total_runs: int,
    curr_run: int,
    bar_length: int,
    avg_time_per_run: float,
    overwrite_output: bool,
) -> None:
    """
    Prints a dynamic progress bar to the terminal.

    Args:
        total_runs: Total number of runs to execute.
        curr_run: Current run index.
        bar_length: Length of the progress bar in characters.
        avg_time_per_run: Average time taken per run, used to estimate remaining time.
        overwrite_output: Controls how configurations are printed:
            - **True**: Overwrites the existing progress bar
            - **False**: Writes a new progress bar
    """
    remaining_time = avg_time_per_run * (total_runs - curr_run)
    formatted_time = format_time(remaining_time)

    # Print progress after each run
    progress = curr_run / total_runs
    filled_length = int(bar_length * progress)
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    if overwrite_output:
        sys.stdout.write("\033[1A")  # Move cursor up 1 line
        sys.stdout.write("\033[2K\r")  # Clear line
        sys.stdout.write(
            f"Progress: [{bar}] {progress * 100:.2f}% | Estimated time remaining: {formatted_time}\n"
        )
        sys.stdout.flush()

    else:
        print(
            f"Progress: [{bar}] {progress * 100:.2f}% | Estimated time remaining: {formatted_time}"
        )


def print_config(
    total_configs: int | None, curr_config: int, c: Any, overwrite_output: bool
) -> None:
    """
    Prints the current configuration being profiled.

    Args:
        total_configs: Total number of configurations.
        curr_config: Current configuration index.
        c: The current configuration parameters.
        overwrite_output: Controls how configurations are printed:
            - **True**: The configuration is updated in-place
            - **False**: Each configuration is printed on a new line
    """
    config_string = (
        f"{curr_config}/{total_configs}"
        if total_configs is not None
        else f"{curr_config}"
    )
    if overwrite_output:
        sys.stdout.write("\033[2F")  # Move cursor up two lines
        sys.stdout.write("\033[2K\r")  # Clear line
        sys.stdout.write(f"Config {config_string}: {str(list(map(str, c)))}\n\n")
        sys.stdout.flush()

    else:
        print_header(f"Config {config_string}: {str(list(map(str, c)))}")


def batched(iterable: Any, n: int) -> Iterator[tuple[Any, ...]]:
    """
    Batch an iterable into tuples of size n.

    This is a minimal backport of itertools.batched for Python 3.10 and 3.11,
    where the standard library implementation is not available.
    """
    if n < 1:
        raise ValueError("n must be atleast 1")

    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


class LogParser:
    """
    Base class for parsing the log files
    """

    def parse_logs(self, log_file_path: str) -> dict[str, list[str]]:
        """
        Parses the log file and returns a list of log entries.

        Args:
            log_file_path: Path to the log file.
        """
        return {}


class NCULogParser(LogParser):
    """
    Parse NCU log file.
    """

    def parse_logs(self, log_file_path: str) -> dict[str, list[str]]:
        """
        Parses the NCU log file and returns a dictionary of log entries categorized by their type.

        Args:
            log_file_path: Path to the NCU log file.
        """
        # Dictionary to categorize logs by their category
        log_entries: dict[str, list[str]] = {"ERROR": [], "PROF": [], "WARNING": []}

        # Pattern for ==ERROR== messages
        error_pattern = re.compile(r"^==ERROR==\s+(.*)$")
        # Pattern for ==PROF== messages
        prof_pattern = re.compile(r"^==PROF==\s+(.*)$")
        # Pattern for ==WARNING== messages
        warning_pattern = re.compile(r"^==WARNING==\s+(.*)$")

        with open(log_file_path, "r") as file:
            for line in file:
                line = line.strip()
                if error_match := error_pattern.match(line):
                    log_entries["ERROR"].append(error_match.group(1))
                elif prof_match := prof_pattern.match(line):
                    log_entries["PROF"].append(prof_match.group(1))
                elif warning_match := warning_pattern.match(line):
                    log_entries["WARNING"].append(warning_match.group(1))

        return log_entries

    def get_logs(self, log_file_path: str, category: str) -> list[str]:
        """
        Returns log entries of a specific category from the NCU log file.

        Args:
            log_file_path: Path to the NCU log file.
            category: Category of logs (e.g., "ERROR", "PROF").
        """
        logs = self.parse_logs(log_file_path)
        return logs.get(category, [])


def format_ncu_error_message(context: NCUErrorContext) -> str:
    """
    Format NCU error context into user-friendly error message.

    Args:
        context: The error context containing all relevant information.
    """

    INVALID_METRIC_ERROR_HINT = "Failed to find metric"

    # FIXME: To support multiple metrics in future, parse error message itself to extract the invalid metric name and display appropriate messages.
    message_parts = ["PROFILING FAILED \nErrors:"]

    if context.errors and INVALID_METRIC_ERROR_HINT in context.errors[0]:
        message_parts.append(
            get_metrics_error_message(
                context.metrics, error_type=MetricErrorType.INVALID
            )
        )
    else:
        message_parts.append("\n".join(f"- {error}" for error in context.errors))

    message_parts.append(
        f"\nRefer Nsight Compute CLI log file: {context.log_file_path} for more details."
    )

    return "\n".join(message_parts)


def find_external_stacklevel() -> int:
    """
    Find the stacklevel corresponding to the first frame outside the nsight package.
    This is equivalent to warnings.warn()'s skip_file_prefixes parameter, which was introduced
    in Python 3.12 .

    Returns:
        int: The stacklevel (1-based, as expected by warnings.warn).
    """
    try:
        import nsight

        pkg_dir = os.path.dirname(os.path.abspath(nsight.__file__))
    except Exception:
        pkg_dir = None

    # Iterate over stack frames
    for level, frame_info in enumerate(inspect.stack(), start=1):
        frame_filename = os.path.abspath(frame_info.filename)
        if pkg_dir is None or not frame_filename.startswith(pkg_dir):
            return level

    # Fallback: if all frames are inside the package, use the topmost one
    return 1


def is_scalar(config: Any) -> bool:
    """Return True if x is a scalar (not a sequence)."""
    return isinstance(config, str) or not isinstance(config, Sequence)
