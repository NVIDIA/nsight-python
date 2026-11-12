# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import nsight
from nsight import exceptions, tools_manager
from nsight.collection import _compute_sanitizer_attach, compute_sanitizer

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="Compute Sanitizer is Linux-only"
)


def test_build_attach_command() -> None:
    attach = compute_sanitizer._build_attach_command(
        "/usr/bin/compute-sanitizer", 51000, "/tmp/sanitizer.log"
    )

    assert attach[0] == "/usr/bin/compute-sanitizer"
    assert attach[attach.index("--mode") + 1] == "attach"
    assert attach[attach.index("--port") + 1] == "51000"
    assert attach[attach.index("--max-connections") + 1] == "1"
    assert attach[attach.index("--nvtx") + 1] == "yes"
    assert attach[attach.index("--log-file") + 1] == "/tmp/sanitizer.log"
    assert "--save" not in attach
    assert "--xml" not in attach


def test_tool_and_arguments_reach_attach_command() -> None:
    options = nsight.ComputeSanitizerOptions(
        tool=nsight.ComputeSanitizerTool.RACECHECK,
        tool_args=["--racecheck-report", "all"],
    )
    attach = compute_sanitizer._build_attach_command(
        "/usr/bin/compute-sanitizer",
        51000,
        "/tmp/sanitizer.log",
        tool=options.tool,
        tool_args=options.tool_args,
    )

    assert attach[attach.index("--tool") + 1] == "racecheck"
    assert attach[attach.index("--racecheck-report") + 1] == "all"
    assert attach[-1] == "python"


def test_tool_must_be_enum_member() -> None:
    with pytest.raises(TypeError, match="ComputeSanitizerTool member"):
        nsight.ComputeSanitizerOptions(tool="racecheck")  # type: ignore[arg-type]


def test_tool_arguments_cannot_override_connection_or_output() -> None:
    for argument in (
        "--tool",
        "--port=42",
        "--nvtx-include",
        "--log-file",
        "--save",
        "--xml",
        "--",
    ):
        with pytest.raises(ValueError, match="cannot override"):
            nsight.ComputeSanitizerOptions(tool_args=(argument,))


@pytest.mark.parametrize("library_dir", [".", "libs"])  # type: ignore[untyped-decorator]
def test_find_collection_library_bundled_with_frontend(
    tmp_path: Path, library_dir: str
) -> None:
    frontend = tmp_path / "release" / "compute-sanitizer"
    collection = frontend.parent / library_dir / "libsanitizer-collection.so"
    frontend.parent.mkdir(parents=True)
    collection.parent.mkdir(parents=True, exist_ok=True)
    frontend.touch()
    collection.touch()
    collection.with_name("libsanitizer-public.so").touch()

    assert compute_sanitizer._find_collection_library(str(frontend)) == str(collection)


def test_find_collection_library_from_matching_build(tmp_path: Path) -> None:
    platform = "linux-desktop-glibc_2_11_3-x64"
    frontend = tmp_path / "host" / platform / "compute-sanitizer"
    collection = tmp_path / "target" / platform / "libsanitizer-collection.so"
    public = collection.with_name("libsanitizer-public.so")
    frontend.parent.mkdir(parents=True)
    collection.parent.mkdir(parents=True)
    frontend.touch()
    collection.touch()
    public.touch()

    assert compute_sanitizer._find_collection_library(str(frontend)) == str(collection)


def test_find_collection_library_requires_matching_public_library(
    tmp_path: Path,
) -> None:
    frontend = tmp_path / "compute-sanitizer"
    frontend.touch()
    frontend.with_name("libsanitizer-collection.so").touch()

    with pytest.raises(
        exceptions.ComputeSanitizerNotAvailableError,
        match="libsanitizer-public.so",
    ):
        compute_sanitizer._find_collection_library(str(frontend))


def test_configure_injection_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = [
        compute_sanitizer._PORT_BASE_ENV_VAR,
        compute_sanitizer._PORT_RANGE_BEGIN_ENV_VAR,
        compute_sanitizer._PORT_RANGE_END_ENV_VAR,
        compute_sanitizer._TRANSPORT_ENV_VAR,
        "NVTX_INJECTION64_PATH",
    ]
    for name in names:
        monkeypatch.delenv(name, raising=False)

    compute_sanitizer._configure_injection_environment(
        51000,
        "/build/libsanitizer-collection.so",
    )

    assert os.environ[compute_sanitizer._PORT_BASE_ENV_VAR] == "51000"
    assert os.environ[compute_sanitizer._PORT_RANGE_BEGIN_ENV_VAR] == "0"
    assert os.environ[compute_sanitizer._PORT_RANGE_END_ENV_VAR] == "0"
    assert os.environ[compute_sanitizer._TRANSPORT_ENV_VAR] == "tcp"
    assert os.environ["NVTX_INJECTION64_PATH"] == "/build/libsanitizer-collection.so"


def test_rejects_activation_after_cuda_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cuda.bindings import driver

    monkeypatch.setattr(
        driver,
        "cuCtxGetCurrent",
        MagicMock(return_value=(driver.CUresult.CUDA_SUCCESS, None)),
    )

    with pytest.raises(
        exceptions.ProfilerException,
        match="before CUDA initialization",
    ):
        compute_sanitizer._ensure_cuda_not_initialized()


def test_init_self_attaches_without_relaunch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = object()
    frontend = MagicMock()
    monkeypatch.setattr(compute_sanitizer, "_backend_handle", None)
    monkeypatch.setattr(compute_sanitizer, "_frontend_process", None)
    monkeypatch.setattr(compute_sanitizer, "_log_path", None)
    monkeypatch.setattr(
        compute_sanitizer,
        "_find_executable",
        MagicMock(return_value="/build/compute-sanitizer"),
    )
    monkeypatch.setattr(compute_sanitizer, "_check_version", MagicMock())
    monkeypatch.setattr(
        compute_sanitizer,
        "_find_collection_library",
        MagicMock(return_value="/build/libsanitizer-collection.so"),
    )
    monkeypatch.setattr(
        compute_sanitizer, "_allocate_port", MagicMock(return_value=51000)
    )
    log_path = Path("/build/sanitizer.log")
    allocate_log = MagicMock(return_value=log_path)
    monkeypatch.setattr(
        compute_sanitizer,
        "_allocate_log_path",
        allocate_log,
    )
    configure = MagicMock()
    load = MagicMock(return_value=handle)
    start = MagicMock(return_value=frontend)
    monkeypatch.setattr(
        compute_sanitizer, "_configure_injection_environment", configure
    )
    monkeypatch.setattr(compute_sanitizer, "_ensure_cuda_not_initialized", MagicMock())
    monkeypatch.setattr(compute_sanitizer, "_load_backend", load)
    monkeypatch.setattr(compute_sanitizer, "_start_attach_helper", start)

    options = nsight.ComputeSanitizerOptions(
        log_path=log_path,
        tool=nsight.ComputeSanitizerTool.RACECHECK,
        tool_args=("--racecheck-report", "all"),
    )
    compute_sanitizer.init(options)

    configure.assert_called_once_with(51000, "/build/libsanitizer-collection.so")
    allocate_log.assert_called_once_with(log_path)
    load.assert_called_once_with("/build/libsanitizer-collection.so")
    start.assert_called_once_with(
        "/build/compute-sanitizer",
        51000,
        log_path,
        options,
    )
    assert compute_sanitizer._backend_handle is handle
    assert compute_sanitizer._frontend_process is frontend
    assert compute_sanitizer._log_path == log_path


def test_init_restores_injection_environment_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(compute_sanitizer._PORT_BASE_ENV_VAR, "old-port")
    monkeypatch.setenv("NVTX_INJECTION64_PATH", "old-collection")
    for name in compute_sanitizer._INJECTION_ENV_VARS:
        if name not in (compute_sanitizer._PORT_BASE_ENV_VAR, "NVTX_INJECTION64_PATH"):
            monkeypatch.delenv(name, raising=False)
    previous = {
        name: os.environ.get(name) for name in compute_sanitizer._INJECTION_ENV_VARS
    }

    monkeypatch.setattr(compute_sanitizer, "_ensure_cuda_not_initialized", MagicMock())
    monkeypatch.setattr(
        compute_sanitizer,
        "_find_executable",
        MagicMock(return_value="/build/compute-sanitizer"),
    )
    monkeypatch.setattr(compute_sanitizer, "_check_version", MagicMock())
    monkeypatch.setattr(
        compute_sanitizer,
        "_find_collection_library",
        MagicMock(return_value="/build/libsanitizer-collection.so"),
    )
    monkeypatch.setattr(
        compute_sanitizer, "_allocate_port", MagicMock(return_value=51000)
    )
    monkeypatch.setattr(
        compute_sanitizer,
        "_allocate_log_path",
        MagicMock(return_value=tmp_path / "sanitizer.log"),
    )

    for failure in ("load", "attach"):
        monkeypatch.setattr(compute_sanitizer, "_backend_handle", None)
        monkeypatch.setattr(compute_sanitizer, "_frontend_process", None)
        monkeypatch.setattr(compute_sanitizer, "_log_path", None)
        monkeypatch.setattr(
            compute_sanitizer,
            "_load_backend",
            MagicMock(
                side_effect=RuntimeError("load failed") if failure == "load" else None,
                return_value=object(),
            ),
        )
        monkeypatch.setattr(
            compute_sanitizer,
            "_start_attach_helper",
            MagicMock(
                side_effect=(
                    RuntimeError("attach failed") if failure == "attach" else None
                )
            ),
        )

        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            compute_sanitizer.init()

        assert {
            name: os.environ.get(name) for name in compute_sanitizer._INJECTION_ENV_VARS
        } == previous


def test_sanitizer_activation_and_annotations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    init = MagicMock()
    monkeypatch.setattr(tools_manager, "_active_tool", None)
    monkeypatch.setattr(compute_sanitizer, "init", init)
    monkeypatch.setattr(compute_sanitizer, "_backend_handle", object())
    frontend = MagicMock()
    frontend.poll.return_value = None
    monkeypatch.setattr(compute_sanitizer, "_frontend_process", frontend)
    log = tmp_path / "sanitizer.log"
    monkeypatch.setattr(compute_sanitizer, "_log_path", log)
    calls: list[int] = []

    assert nsight.get_active_tool() is None
    with pytest.raises(exceptions.ProfilerException, match="not active"):
        nsight.sanitize.log_path()
    assert calls == []

    options = nsight.ComputeSanitizerOptions(
        log_path=log,
        tool=nsight.ComputeSanitizerTool.RACECHECK,
        tool_args=("--racecheck-report", "all"),
    )
    nsight.activate(nsight.Tool.COMPUTE_SANITIZER, options=options)
    # Re-activating is a no-op that warns, whether or not options are passed.
    with pytest.warns(RuntimeWarning, match="already active"):
        nsight.activate(nsight.Tool.COMPUTE_SANITIZER)
    with pytest.warns(RuntimeWarning, match="already active"):
        nsight.activate(
            nsight.Tool.COMPUTE_SANITIZER,
            options=nsight.ComputeSanitizerOptions(
                tool=nsight.ComputeSanitizerTool.MEMCHECK
            ),
        )
    for value in (7, 8):
        with nsight.annotate("target"):
            calls.append(value)
    assert nsight.sanitize.log_path() == log

    init.assert_called_once_with(options)
    assert calls == [7, 8]
    frontend.poll.assert_called()


def test_sanitizer_rejects_nested_annotations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tools_manager, "_active_tool", nsight.Tool.COMPUTE_SANITIZER)
    region = nsight.annotate("target")

    with region:
        with pytest.raises(ValueError, match="Nested annotations are not supported"):
            with region:
                pass

    # The same name remains usable in a later, non-nested range.
    with region:
        pass


def test_helper_detects_when_port_starts_and_stops_listening() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

        assert not _compute_sanitizer_attach._port_is_listening(port)

        listener.listen()
        assert _compute_sanitizer_attach._port_is_listening(port)

    assert not _compute_sanitizer_attach._port_is_listening(port)


def test_helper_recognizes_current_parent() -> None:
    assert _compute_sanitizer_attach._parent_is_alive(os.getpid())


def test_supported_compute_sanitizer_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        MagicMock(
            return_value=SimpleNamespace(
                returncode=0,
                stdout="NVIDIA Compute Sanitizer\nVersion 2026.4.0.0",
                stderr="",
            )
        ),
    )
    compute_sanitizer._check_version("/usr/bin/compute-sanitizer")


@pytest.mark.parametrize("version", ["2026.3.0.0", "2022.4.1"])  # type: ignore[untyped-decorator]
def test_rejects_old_compute_sanitizer_version(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        MagicMock(
            return_value=SimpleNamespace(
                returncode=0, stdout=f"Version {version}", stderr=""
            )
        ),
    )
    with pytest.raises(
        exceptions.ComputeSanitizerNotSupportedError,
        match="2026.4.0.0 or later is required",
    ):
        compute_sanitizer._check_version("/usr/bin/compute-sanitizer")


def test_rejects_unknown_compute_sanitizer_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        MagicMock(return_value=SimpleNamespace(returncode=0, stdout="", stderr="")),
    )
    with pytest.raises(
        exceptions.ComputeSanitizerNotSupportedError,
        match="Could not determine",
    ):
        compute_sanitizer._check_version("/usr/bin/compute-sanitizer")


def test_log_path_option_and_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    explicit = tmp_path / "explicit.log"
    assert compute_sanitizer._allocate_log_path(explicit) == explicit
    explicit.touch()
    with pytest.raises(exceptions.ProfilerException, match="already exists"):
        compute_sanitizer._allocate_log_path(explicit)
    default = compute_sanitizer._allocate_log_path()
    assert default.parent == tmp_path
    assert default.name.startswith("compute-sanitizer-output-")
    assert default.suffix == ".log"
