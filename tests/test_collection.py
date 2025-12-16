# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from typing import Any, Dict
from unittest.mock import MagicMock, call, patch

import pytest

from nsight import collection, exceptions


@patch("subprocess.run")
def test_launch_ncu_runs_with_ncu_available(mock_run: MagicMock) -> None:
    # Simulate "ncu --version" runs successfully, and so does the main profiling command
    mock_run.side_effect = [None, None]  # ncu --version  # actual ncu profiling run

    collection.ncu.launch_ncu(
        "report.ncu-rep",
        "func_name",
        metrics=["sm__cycles_elapsed.avg"],
        cache_control="all",
        clock_control="base",
        replay_mode="kernel",
        verbose=True,
    )

    expected_calls = [
        call(
            ["ncu", "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ),
        call(
            pytest.helpers.mock_any_command_string(),
            shell=True,
            check=True,
            env=pytest.helpers.env_contains({"NSPY_NCU_PROFILE": "func_name"}),
        ),
    ]

    # Check if subprocess.run was called twice
    assert mock_run.call_count == 2
    assert "--nvtx-include" in mock_run.call_args_list[1].args[0]


@patch("subprocess.run")
def test_launch_ncu_falls_back_without_ncu(mock_run: MagicMock) -> None:
    # Simulate "ncu --version" fails, fallback to run the script and raise NCUNotAvailableError
    mock_run.side_effect = [
        FileNotFoundError(),  # ncu --version
        None,  # fallback to plain script run
    ]

    with pytest.raises(exceptions.NCUNotAvailableError) as exc_info:
        collection.ncu.launch_ncu(
            "report.ncu-rep",
            "func_name",
            metrics=["metric"],
            cache_control="all",
            clock_control="base",
            replay_mode="kernel",
            verbose=False,
        )

    # Verify the exception message
    assert "Nsight Compute CLI (ncu) is not available" in str(exc_info.value)

    assert mock_run.call_count == 2
    assert sys.executable in mock_run.call_args_list[1].args[0]


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
            subset: Dict[str, str] = self
            return all(item in other.items() for item in subset.items())

    pytest.helpers = type("helpers", (), {})()

    def mock_any_command_string() -> Matcher:
        return Matcher("any-ncu-command")

    def env_contains(expected_subset: Dict[str, str]) -> EnvMatcher:
        return EnvMatcher(expected_subset)

    pytest.helpers.mock_any_command_string = mock_any_command_string
    pytest.helpers.env_contains = env_contains
