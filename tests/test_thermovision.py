# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any
from unittest.mock import patch

import torch

import nsight

# ============================================================================
# Thermovision integration tests
# ============================================================================


@nsight.analyze.kernel(configs=[(1024,)], runs=1, thermal_control=True, output="quiet")
def thermo_kernel(n: int) -> None:
    """Dummy Test kernel with thermovision enabled."""
    pass


def test_thermovision_module_with_thermal_waiting() -> None:
    """Test thermovision when GPU needs cooling (mocked hot GPU scenario)."""

    # Store original environment variable value
    original_env = os.environ.get("NSPY_NCU_PROFILE")

    try:
        # Set environment variable to prevent subprocess spawning
        os.environ["NSPY_NCU_PROFILE"] = "thermo_kernel"

        tlimit_calls = 0
        temp_calls = 0

        def get_tlimit(handle: Any) -> int:
            nonlocal tlimit_calls
            tlimit_calls += 1
            if tlimit_calls == 1:
                return 1  # First call: extremely low (triggers thermal waiting)
            else:
                return 100  # Subsequent calls: very high (exits thermal waiting)

        def get_temp(handle: Any) -> int:
            nonlocal temp_calls
            temp_calls += 1
            return max(10, 80 - temp_calls * 10)  # Temperature gradually decreases

        with (
            patch("nsight.thermovision.init", return_value=True),
            patch("nsight.thermovision.get_gpu_tlimit", side_effect=get_tlimit),
            patch("nsight.thermovision.get_gpu_temp", side_effect=get_temp),
            patch("nsight.thermovision.time.sleep") as mock_sleep,
            patch("os._exit"),
        ):

            thermo_kernel()

            # Verify thermal management caused sleep calls (GPU cooling simulation)
            assert (
                mock_sleep.called
            ), "Thermal management trigger GPU cooling (sleep) as expected"

    finally:
        # Restore original environment variable
        if original_env is None:
            os.environ.pop("NSPY_NCU_PROFILE", None)
        else:
            os.environ["NSPY_NCU_PROFILE"] = original_env


def test_thermovision_module_without_thermal_waiting() -> None:
    """Test thermovision when GPU is already cool (mocked scenario)."""

    # Store original environment variable value
    original_env = os.environ.get("NSPY_NCU_PROFILE")

    try:
        # Set environment variable to prevent subprocess spawning
        os.environ["NSPY_NCU_PROFILE"] = "thermo_kernel"

        def get_tlimit(handle: Any) -> int:
            return 100  # Always high, no waiting needed

        with (
            patch("nsight.thermovision.init", return_value=True),
            patch("nsight.thermovision.get_gpu_tlimit", side_effect=get_tlimit),
            patch("nsight.thermovision.get_gpu_temp", return_value=10),
            patch("nsight.thermovision.time.sleep") as mock_sleep,
            patch("os._exit"),
        ):

            thermo_kernel()

            # Verify thermal management did not cause sleep calls
            assert (
                not mock_sleep.called
            ), "Thermal management should not have triggered sleep"

    finally:
        # Restore original environment variable
        if original_env is None:
            os.environ.pop("NSPY_NCU_PROFILE", None)
        else:
            os.environ["NSPY_NCU_PROFILE"] = original_env
