# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
from typing import Any

import pytest
import torch


def get_cuda_dev_cc_major(device_id: int) -> Any:
    props = torch.cuda.get_device_properties(device_id)
    return props.major


def test_00_minimal() -> None:
    minimal = importlib.import_module("examples.00_minimal")
    minimal.main()


def test_01_compare_throughput() -> None:
    compare_throughput = importlib.import_module("examples.01_compare_throughput")
    compare_throughput.main()


def test_02_parameter_sweep() -> None:
    parameter_sweep = importlib.import_module("examples.02_parameter_sweep")
    parameter_sweep.main()


def test_03_custom_metrics() -> None:
    custom_metrics = importlib.import_module("examples.03_custom_metrics")
    custom_metrics.main()


def test_04_multi_parameter() -> None:
    multi_parameter = importlib.import_module("examples.04_multi_parameter")
    multi_parameter.main()


def test_05_subplots() -> None:
    subplots = importlib.import_module("examples.05_subplots")
    subplots.main()


def test_06_plot_customization() -> None:
    plot_customization = importlib.import_module("examples.06_plot_customization")
    plot_customization.main()


def test_07_triton_minimal() -> None:
    pytest.importorskip("triton")
    triton_minimal = importlib.import_module("examples.07_triton_minimal")
    triton_minimal.main()


def test_08_multiple_metrics() -> None:
    multiple_metrics = importlib.import_module("examples.08_multiple_metrics")
    multiple_metrics.main()


def test_09_advanced_metric_custom() -> None:
    advanced_custom = importlib.import_module("examples.09_advanced_metric_custom")
    advanced_custom.main()


def test_10_combine_kernel_metrics() -> None:
    combine_metrics = importlib.import_module("examples.10_combine_kernel_metrics")
    combine_metrics.main()


def test_11_output_csv() -> None:
    output_csv = importlib.import_module("examples.11_output_csv")
    output_csv.main()


# skip cuTile test on CC 9.x as Cuda Toolkit 13.2 used for testing does not support cuda-tile on CC 9.x (Hopper)
@pytest.mark.skipif(
    get_cuda_dev_cc_major(0) == 9,
    reason="cuda-tile not supported on CC 9.x in CUDA Toolkit 13.2",
)  # type: ignore[untyped-decorator]
def test_12_cutile() -> None:
    cutile = importlib.import_module("examples.12_cutile")
    cutile.main()
