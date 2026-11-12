# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration coverage requiring two physical GPUs in one profiling process."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from cuda.bindings import runtime
from cuda.core import Device, LaunchConfig, Program, launch

import nsight

_KERNELS = """
extern "C" __global__ void on_gpu_zero() {}
extern "C" __global__ void on_gpu_one() {}
"""


def test_variable_kernel_counts_across_devices(tmp_path: Path) -> None:
    # Initialize the default tool before touching CUDA, including in lazy mode.
    if nsight.get_active_tool() is None:
        nsight.activate(nsight.Tool.NCU)
    status, device_count = runtime.cudaGetDeviceCount()
    if status != runtime.cudaError_t.cudaSuccess or device_count < 2:
        pytest.skip("requires at least two CUDA devices")

    status, original_device = runtime.cudaGetDevice()
    assert status == runtime.cudaError_t.cudaSuccess
    configs = [(0, 2), (1, 4), (0, 3), (1, 1)]
    resources: list[tuple[Any, Any, Any]] = []
    try:
        for device_index in range(2):
            device = Device(device_index)
            device.set_current()
            stream = device.create_stream()
            module = Program(_KERNELS, "c++").compile("cubin")
            kernel = module.get_kernel(
                "on_gpu_zero" if device_index == 0 else "on_gpu_one"
            )
            resources.append((stream, module, kernel))

        @nsight.experimental.collect(scope=nsight.CollectionScope.ANNOTATION)
        def active_device(annotation_name: str, *args: Any) -> int:
            status, actual_device = runtime.cudaGetDevice()
            assert status == runtime.cudaError_t.cudaSuccess
            return int(actual_device)

        @nsight.analyze.kernel(
            configs=configs,
            runs=2,
            thermal_mode="off",
            verbosity=nsight.VerbosityLevel.SILENT,
            combine_kernel_metrics=lambda left, right: left + right,
            info_collectors=[active_device],
            output_prefix=str(tmp_path / "devices-"),
            output_csv=True,
        )
        def variable(device_index: int, count: int) -> None:
            Device(device_index).set_current()
            stream, _module, kernel = resources[device_index]
            with nsight.annotate("work"):
                for _ in range(count):
                    launch(stream, LaunchConfig(grid=1, block=1), kernel)
            stream.sync()

        result = variable().to_dataframe()
        raw = pd.read_csv(next(tmp_path.glob("devices-profiled_data-*.csv")))
        expected = [config for config in configs for _ in range(2)]
        assert list(zip(raw.device_index, raw["count"])) == expected
        expected_names = [
            "|".join(["on_gpu_zero" if dev == 0 else "on_gpu_one"] * count)
            for dev, count in expected
        ]
        assert raw.Kernel.tolist() == expected_names
        assert raw.active_device.tolist() == [dev for dev, _count in expected]
        assert (raw.Value > 0).all()
        assert result.NumRuns.tolist() == [2] * len(configs)
        assert list(zip(result.device_index, result["count"])) == configs
        assert result.active_device_Min.tolist() == [dev for dev, _ in configs]
        assert result.active_device_Max.tolist() == [dev for dev, _ in configs]
        assert result.AvgValue.tolist() == pytest.approx(
            [raw.Value.iloc[i : i + 2].mean() for i in range(0, len(expected), 2)]
        )
    finally:
        (status,) = runtime.cudaSetDevice(original_device)
        assert status == runtime.cudaError_t.cudaSuccess
