# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest


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
