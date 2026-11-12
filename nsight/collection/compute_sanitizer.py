# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute Sanitizer self-attach integration.

Activation loads the collection backend into the current pre-CUDA process and
starts one attach frontend when the backend begins listening. The user's Python
script is never relaunched.
"""

import ctypes
import enum
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from nsight import exceptions, utils


class ComputeSanitizerTool(str, enum.Enum):
    """Checking tools selectable with Compute Sanitizer."""

    MEMCHECK = "memcheck"
    RACECHECK = "racecheck"
    INITCHECK = "initcheck"
    SYNCCHECK = "synccheck"


DEFAULT_TOOL = ComputeSanitizerTool.MEMCHECK
DEFAULT_LAUNCH_TIMEOUT = 30
DEFAULT_ERROR_EXITCODE = 1
# Compute Sanitizer version shipped with CUDA Toolkit 13.5.
MIN_COMPUTE_SANITIZER_VERSION = (2026, 4, 0, 0)
_PORT_BASE_ENV_VAR = "NV_SANITIZER_INJECTION_PORT_BASE"
_PORT_RANGE_BEGIN_ENV_VAR = "NV_SANITIZER_INJECTION_PORT_RANGE_BEGIN"
_PORT_RANGE_END_ENV_VAR = "NV_SANITIZER_INJECTION_PORT_RANGE_END"
_TRANSPORT_ENV_VAR = "NV_SANITIZER_INJECTION_TRANSPORT_TYPE"
_INJECTION_ENV_VARS = (
    _PORT_BASE_ENV_VAR,
    _PORT_RANGE_BEGIN_ENV_VAR,
    _PORT_RANGE_END_ENV_VAR,
    _TRANSPORT_ENV_VAR,
    "NVTX_INJECTION64_PATH",
)

_backend_handle: ctypes.CDLL | None = None
_frontend_process: subprocess.Popen[Any] | None = None
_log_path: Path | None = None


@dataclass(frozen=True)
class ComputeSanitizerOptions:
    """Settings for ``nsight.activate(Tool.COMPUTE_SANITIZER, options=...)``."""

    log_path: str | os.PathLike[str] | None = None
    """Destination for live text diagnostics. If omitted, a unique filename is
    chosen in the system temporary directory. The parent directory must exist,
    and the log path must not already exist.
    """

    tool: ComputeSanitizerTool = DEFAULT_TOOL
    """Checking tool to run. Defaults to :attr:`ComputeSanitizerTool.MEMCHECK`.
    """

    tool_args: Sequence[str] = ()
    """Additional CLI argument tokens for the selected tool. Pass each option
    and value separately, for example ``("--racecheck-report", "all")``; this
    is not a shell command string. Connection, NVTX, and log output options
    remain under Nsight Python's control and cannot be overridden here.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.tool, ComputeSanitizerTool):
            raise TypeError("tool must be a ComputeSanitizerTool member")
        if isinstance(self.tool_args, str) or any(
            not isinstance(arg, str) for arg in self.tool_args
        ):
            raise TypeError("tool_args must be a sequence of argument strings")
        arguments = tuple(self.tool_args)
        reserved = {
            "--mode",
            "--port",
            "--max-connections",
            "--launch-timeout",
            "--tool",
            "--error-exitcode",
            "--nvtx",
            "--nvtx-include",
            "--log-file",
            "--save",
            "--xml",
            "--print-limit",
        }
        if "--" in arguments or any(
            arg.split("=", 1)[0] in reserved for arg in arguments
        ):
            raise ValueError(
                "tool_args cannot override Nsight Python's sanitizer options"
            )
        object.__setattr__(self, "tool_args", arguments)


def _nvtx_include_args(domain: str = utils.NVTX_DOMAIN) -> list[str]:
    """Return filters for all push/pop and start/end ranges in ``domain``."""
    escaped_domain = re.escape(domain)
    return [
        "--nvtx-include",
        rf"regex:^{escaped_domain}$@.*/",
        "--nvtx-include",
        rf"regex:^{escaped_domain}$@.*",
    ]


def _build_attach_command(
    executable: str,
    port: int,
    log_path: str,
    *,
    tool: ComputeSanitizerTool = DEFAULT_TOOL,
    tool_args: Sequence[str] = (),
    launch_timeout: int = DEFAULT_LAUNCH_TIMEOUT,
    error_exitcode: int = DEFAULT_ERROR_EXITCODE,
) -> list[str]:
    return [
        executable,
        "--mode",
        "attach",
        "--port",
        str(port),
        "--max-connections",
        "1",
        "--launch-timeout",
        str(launch_timeout),
        "--tool",
        tool.value,
        "--error-exitcode",
        str(error_exitcode),
        "--nvtx",
        "yes",
        *_nvtx_include_args(),
        "--log-file",
        log_path,
        "--print-limit",
        "0",
        *tool_args,
        # The current Compute Sanitizer parser requires one positional value in
        # attach mode even though the frontend does not launch this command.
        "python",
    ]


def _allocate_port() -> int:
    """Ask the OS for an unused local TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _allocate_log_path(requested_path: str | os.PathLike[str] | None = None) -> Path:
    """Choose a new text log path without overwriting an existing file."""
    if requested_path is not None:
        log = Path(requested_path).expanduser().resolve()
    else:
        log = Path(tempfile.gettempdir()) / (
            f"compute-sanitizer-output-{os.getpid()}-{uuid.uuid4().hex[:8]}.log"
        )

    if not log.parent.is_dir():
        raise exceptions.ProfilerException(
            f"Compute Sanitizer log directory does not exist: {log.parent}"
        )
    if log.exists():
        raise exceptions.ProfilerException(
            f"Compute Sanitizer log path already exists: {log}"
        )
    return log


def _find_executable() -> str:
    executable = shutil.which("compute-sanitizer")
    if executable is None:
        raise exceptions.ComputeSanitizerNotAvailableError(
            "Compute Sanitizer is not available on $PATH. Install a CUDA Toolkit "
            "version that provides the compute-sanitizer executable."
        )
    return executable


def _check_version(executable: str) -> None:
    """Require a Compute Sanitizer version supported by this integration."""
    result = subprocess.run(
        [executable, "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"Version\s+(\d+)\.(\d+)\.(\d+)(?:\.(\d+))?", result.stdout)
    if result.returncode != 0 or match is None:
        raise exceptions.ComputeSanitizerNotSupportedError(
            "Could not determine the Compute Sanitizer version."
        )
    version = tuple(int(component) for component in match.groups(default="0"))
    if version < MIN_COMPUTE_SANITIZER_VERSION:
        minimum = ".".join(
            str(component) for component in MIN_COMPUTE_SANITIZER_VERSION
        )
        found = ".".join(str(component) for component in version)
        raise exceptions.ComputeSanitizerNotSupportedError(
            f"Compute Sanitizer {minimum} or later is required, but found {found}."
        )


def _ensure_cuda_not_initialized() -> None:
    """Reject activation after CUDA initialization without initializing it."""
    from cuda.bindings import driver

    status, _ = driver.cuCtxGetCurrent()
    if status == driver.CUresult.CUDA_ERROR_NOT_INITIALIZED:
        return
    if status == driver.CUresult.CUDA_SUCCESS:
        raise exceptions.ProfilerException(
            "Compute Sanitizer must be activated before CUDA initialization."
        )
    raise exceptions.ProfilerException(
        "Could not determine whether CUDA was already initialized: "
        f"cuCtxGetCurrent returned {status}."
    )


def _find_collection_library(executable: str) -> str:
    """Find the backend in a frontend bundle, CUDA Toolkit, or matching build."""
    frontend = Path(executable).resolve()
    platform_dir = frontend.parent.name
    build_dir = frontend.parent.parent.parent
    candidates = [
        frontend.with_name("libsanitizer-collection.so"),
        frontend.parent / "libs" / "libsanitizer-collection.so",
    ]
    if frontend.parent.name == "bin":
        # CUDA Toolkit installs bin/compute-sanitizer as a shell launcher for
        # ../compute-sanitizer/compute-sanitizer. resolve() follows symlinks
        # such as /usr/local/cuda, but cannot follow a shell launcher's exec.
        candidates.append(
            frontend.parent.parent / "compute-sanitizer" / "libsanitizer-collection.so"
        )
    candidates.append(
        build_dir / "target" / platform_dir / "libsanitizer-collection.so"
    )

    for collection in candidates:
        if (
            collection.is_file()
            and collection.with_name("libsanitizer-public.so").is_file()
        ):
            return str(collection)

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise exceptions.ComputeSanitizerNotAvailableError(
        "Could not find a matching Compute Sanitizer collection backend and "
        f"libsanitizer-public.so. Searched {searched}."
    )


def _configure_injection_environment(port: int, collection: str) -> None:
    os.environ[_PORT_BASE_ENV_VAR] = str(port)
    os.environ[_PORT_RANGE_BEGIN_ENV_VAR] = "0"
    os.environ[_PORT_RANGE_END_ENV_VAR] = "0"
    os.environ[_TRANSPORT_ENV_VAR] = "tcp"
    os.environ["NVTX_INJECTION64_PATH"] = collection


def _load_backend(collection: str) -> ctypes.CDLL:
    try:
        return ctypes.CDLL(
            collection,
            mode=os.RTLD_NOW | os.RTLD_GLOBAL,
        )
    except OSError as exc:
        raise exceptions.ProfilerException(
            f"Failed to load Compute Sanitizer backend {collection}: {exc}"
        ) from exc


def _start_attach_helper(
    executable: str,
    port: int,
    log_path: Path,
    options: ComputeSanitizerOptions,
) -> subprocess.Popen[Any]:
    helper = Path(__file__).with_name("_compute_sanitizer_attach.py")
    attach_command = _build_attach_command(
        executable,
        port,
        str(log_path),
        tool=options.tool,
        tool_args=options.tool_args,
    )
    return subprocess.Popen(
        [
            sys.executable,
            str(helper),
            str(os.getpid()),
            str(port),
            *attach_command,
        ]
    )


def init(options: ComputeSanitizerOptions | None = None) -> None:
    """Self-inject Compute Sanitizer before the process initializes CUDA."""
    global _backend_handle, _frontend_process, _log_path

    options = options or ComputeSanitizerOptions()

    if _backend_handle is not None:
        if _frontend_process is None:
            raise exceptions.ProfilerException(
                "Compute Sanitizer backend loaded, but its attach helper failed "
                "to start; restart the Python process before trying again."
            )
        return
    if os.name != "posix" or not Path("/proc/net/tcp").exists():
        raise exceptions.ComputeSanitizerNotSupportedError(
            "Compute Sanitizer self-attach currently requires Linux with /proc."
        )

    _ensure_cuda_not_initialized()

    executable = _find_executable()
    _check_version(executable)
    collection = _find_collection_library(executable)
    port = _allocate_port()
    log_path = _allocate_log_path(options.log_path)
    previous_environment = {name: os.environ.get(name) for name in _INJECTION_ENV_VARS}
    try:
        _configure_injection_environment(port, collection)

        # The handle must remain alive for the rest of the Python process.
        _backend_handle = _load_backend(collection)
        _log_path = log_path
        _frontend_process = _start_attach_helper(executable, port, log_path, options)
    except BaseException:
        for name, previous_value in previous_environment.items():
            if previous_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous_value
        raise


def get_log_path() -> Path:
    """Return the live log destination once the injected backend is ready."""
    if _backend_handle is None or _frontend_process is None or _log_path is None:
        raise exceptions.ProfilerException(
            "Compute Sanitizer is not initialized. Call "
            "nsight.activate(nsight.Tool.COMPUTE_SANITIZER) before CUDA use."
        )
    frontend_returncode = _frontend_process.poll()
    if frontend_returncode is not None:
        raise exceptions.ProfilerException(
            "Compute Sanitizer frontend exited before CUDA attachment "
            f"with status {frontend_returncode}."
        )
    return _log_path
