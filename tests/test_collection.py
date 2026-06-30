# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from nsight import VerbosityLevel, collection


def func_name(x: int, y: int, z: int) -> None:
    pass


@patch("subprocess.Popen")
def test_launch_ncu_runs_with_ncu_available(mock_popen: MagicMock) -> None:
    mock_popen.return_value = MagicMock()

    target_pid = os.getpid()

    collection.ncu.launch_ncu(
        "report.ncu-rep",
        metrics=["sm__cycles_elapsed.avg"],
        cache_control="all",
        clock_control="base",
        replay_mode="kernel",
        verbosity=VerbosityLevel.SILENT,
    )

    assert mock_popen.call_count == 1
    cmd = mock_popen.call_args_list[0].args[0]
    assert cmd == [
        "ncu",
        "--mode",
        "attach",
        "--process-id",
        str(target_pid),
        "--nvtx-include",
        "regex:nsight-python@.+/",
        "--log-file",
        "report.log",
        "--cache-control",
        "all",
        "--clock-control",
        "base",
        "--replay-mode",
        "kernel",
        "--metrics",
        "sm__cycles_elapsed.avg",
        "-f",
        "-o",
        "report.ncu-rep",
    ]


# Optional: Add helpers if you want to cleanly test env vars or command strings
@pytest.fixture(autouse=True)  # type: ignore[untyped-decorator]
def patch_helpers(monkeypatch: Any) -> None:
    class Matcher(str):
        def __eq__(self, other: object) -> bool:
            return isinstance(other, str) and "ncu" in other

    class EnvMatcher(dict[str, str]):
        def __eq__(self, other: object) -> bool:
            if not isinstance(other, dict):
                return False
            subset: dict[str, str] = self
            return all(item in other.items() for item in subset.items())

    pytest.helpers = type("helpers", (), {})()

    def mock_any_command_string() -> Matcher:
        return Matcher("any-ncu-command")

    def env_contains(expected_subset: dict[str, str]) -> EnvMatcher:
        return EnvMatcher(expected_subset)

    pytest.helpers.mock_any_command_string = mock_any_command_string
    pytest.helpers.env_contains = env_contains
