# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from nsight import VerbosityLevel, collection
from nsight.collection import cupti as cupti_collection


def func_name(x: int, y: int, z: int) -> None:
    pass


@patch("shutil.which", return_value="resolved-ncu")
@patch("subprocess.Popen")
def test_launch_ncu_runs_with_ncu_available(
    mock_popen: MagicMock, mock_which: MagicMock
) -> None:
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
        "resolved-ncu",
        "--mode",
        "attach",
        "--process-id",
        str(target_pid),
        "--nvtx-include",
        "regex:nsight-python@.+",
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


# ==============================================================================
# CUPTI kernel attribution
#
# _attribute_kernels is pure, so these run without a GPU or CUPTI installed.
# ==============================================================================


def _marker(marker_id: int, timestamp: int, is_start: bool, name: str) -> Any:
    return cupti_collection._MarkerRecord(marker_id, timestamp, is_start, name)


def test_attribute_kernels_ignores_launches_outside_any_range() -> None:
    markers = [_marker(1, 100, True, "region"), _marker(1, 200, False, "")]
    launches = [
        cupti_collection._LaunchRecord(correlation_id=1, start=50),
        cupti_collection._LaunchRecord(correlation_id=2, start=150),
        cupti_collection._LaunchRecord(correlation_id=3, start=250),
    ]
    kernels = [
        cupti_collection._KernelRecord(correlation_id=i, name=f"k{i}", duration=i)
        for i in (1, 2, 3)
    ]

    assert cupti_collection._attribute_kernels(markers, launches, kernels) == [
        ("region", "k2", 2)
    ]


def test_attribute_kernels_orders_results_by_launch_time() -> None:
    """Position in the result encodes (config, run), so order must be launch order."""
    markers = [_marker(1, 0, True, "region"), _marker(1, 1000, False, "")]
    launches = [
        cupti_collection._LaunchRecord(correlation_id=1, start=300),
        cupti_collection._LaunchRecord(correlation_id=2, start=100),
        cupti_collection._LaunchRecord(correlation_id=3, start=200),
    ]
    # Kernel records deliberately supplied out of order, as CUPTI delivers them.
    kernels = [
        cupti_collection._KernelRecord(correlation_id=3, name="third", duration=3),
        cupti_collection._KernelRecord(correlation_id=1, name="first", duration=1),
        cupti_collection._KernelRecord(correlation_id=2, name="second", duration=2),
    ]

    assert cupti_collection._attribute_kernels(markers, launches, kernels) == [
        ("region", "second", 2),
        ("region", "third", 3),
        ("region", "first", 1),
    ]


def test_attribute_kernels_drops_unclosed_range() -> None:
    """An unclosed range is dropped rather than swallowing every later launch."""
    markers = [_marker(1, 100, True, "region")]
    launches = [cupti_collection._LaunchRecord(correlation_id=7, start=150)]
    kernels = [cupti_collection._KernelRecord(correlation_id=7, name="k", duration=42)]

    assert cupti_collection._attribute_kernels(markers, launches, kernels) == []


def test_attribute_kernels_drops_launch_without_a_kernel_record() -> None:
    markers = [_marker(1, 100, True, "region"), _marker(1, 200, False, "")]
    launches = [cupti_collection._LaunchRecord(correlation_id=7, start=150)]

    assert cupti_collection._attribute_kernels(markers, launches, []) == []


def test_attribute_kernels_pairs_sequential_ranges() -> None:
    """Each launch lands in the range open at the time, not a neighbouring one."""
    markers = []
    launches = []
    kernels = []
    for i in range(3):
        # Range i spans [100i, 100i + 50); the launch sits inside it.
        markers.append(_marker(i, 100 * i, True, f"region_{i}"))
        markers.append(_marker(i, 100 * i + 50, False, ""))
        launches.append(
            cupti_collection._LaunchRecord(correlation_id=i, start=100 * i + 10)
        )
        kernels.append(
            cupti_collection._KernelRecord(
                correlation_id=i, name=f"k{i}", duration=i + 1
            )
        )

    assert cupti_collection._attribute_kernels(markers, launches, kernels) == [
        ("region_0", "k0", 1),
        ("region_1", "k1", 2),
        ("region_2", "k2", 3),
    ]
