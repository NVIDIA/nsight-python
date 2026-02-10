# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any
from unittest.mock import patch

import torch

import nsight
from nsight import collection

# ============================================================================
# Thermovision integration tests
# ============================================================================


@nsight.analyze.kernel(configs=[(1024,)], runs=1, thermal_mode="auto", output="quiet")
def thermo_kernel(n: int) -> None:
    """Dummy Test kernel with thermovision enabled."""
    pass


def test_thermovision_module_with_thermal_waiting() -> None:
    """Test thermovision when GPU needs cooling (mocked hot GPU scenario)."""

    # Store original environment variable value
    original_env = os.environ.get("NSPY_NCU_PROFILE")

    try:
        # Set environment variable to prevent subprocess spawning
        os.environ["NSPY_NCU_PROFILE"] = collection.ncu.get_signature(
            thermo_kernel, [(1024,)]
        )

        tlimit_calls = 0
        temp_calls = 0

        def get_tlimit() -> int:
            nonlocal tlimit_calls
            tlimit_calls += 1
            if tlimit_calls == 1:
                return 1  # First call: extremely low (triggers thermal waiting)
            else:
                return 100  # Subsequent calls: very high (exits thermal waiting)

        def get_temp() -> int:
            nonlocal temp_calls
            temp_calls += 1
            return max(10, 80 - temp_calls * 10)  # Temperature gradually decreases

        with (
            patch("nsight.thermovision.ThermalController.init", return_value=True),
            patch(
                "nsight.thermovision.ThermalController._get_gpu_tlimit",
                side_effect=get_tlimit,
            ),
            patch(
                "nsight.thermovision.ThermalController._get_gpu_temp",
                side_effect=get_temp,
            ),
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
        os.environ["NSPY_NCU_PROFILE"] = collection.ncu.get_signature(
            thermo_kernel, [(1024,)]
        )

        def get_tlimit() -> int:
            return 100  # Always high, no waiting needed

        with (
            patch("nsight.thermovision.ThermalController.init", return_value=True),
            patch(
                "nsight.thermovision.ThermalController._get_gpu_tlimit",
                side_effect=get_tlimit,
            ),
            patch(
                "nsight.thermovision.ThermalController._get_gpu_temp", return_value=10
            ),
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


def test_adaptive_thermal_control_increase() -> None:
    """Test that adaptive mode increases thermal_cont when GPU heats quickly (few iterations)."""
    from nsight.thermovision import ThermalController

    # Create controller in adaptive mode
    controller = ThermalController(thermal_mode="auto", verbose=False)

    # Verify initial state
    assert controller.adaptive_mode is True
    assert controller.thermal_cont == 20  # Default for adaptive
    assert controller.thermal_wait == 10

    call_count = 0

    def mock_tlimit() -> int:
        """Simulate GPU heating cycle with FEW iterations (should increase thermal_cont)."""
        nonlocal call_count
        call_count += 1

        # Simulate: Start → 3 iterations → cooling
        if call_count == 1:
            return 25  # Cool enough to start counting
        elif call_count in [2, 3, 4]:  # 3 iterations
            return 15  # Still above thermal_wait
        elif call_count == 5:
            return 8  # Drop below thermal_wait=10 → trigger cooling
        elif call_count <= 10:
            return 8  # Cooling...
        else:
            return 25  # Cooled back up

    def mock_temp() -> int:
        return 70  # Constant temp for simplicity

    # Patch instance methods
    controller._get_gpu_tlimit = mock_tlimit  # type: ignore[method-assign]
    controller._get_gpu_temp = mock_temp  # type: ignore[method-assign]

    # Simulate thermal guard calls
    for _ in range(15):
        controller.throttle_guard()

    # After cycle with 3 iterations (< target_min=5), thermal_cont should increase
    # Initial: 20, After: 20 + 3 = 23
    assert (
        controller.thermal_cont == 23
    ), f"Expected thermal_cont=23, got {controller.thermal_cont}"


def test_adaptive_thermal_control_decrease() -> None:
    """Test that adaptive mode decreases thermal_cont when GPU heats slowly (many iterations)."""
    from nsight.thermovision import ThermalController

    # Create controller in adaptive mode
    controller = ThermalController(thermal_mode="auto", verbose=False)

    call_count = 0

    def mock_tlimit() -> int:
        """Simulate GPU heating cycle with MANY iterations (should decrease thermal_cont)."""
        nonlocal call_count
        call_count += 1

        # Simulate: Start → 55 iterations → cooling
        if call_count == 1:
            return 25  # Start counting
        elif call_count <= 56:  # 55 iterations (2-56)
            return 12  # Above thermal_wait=10, keeps running
        elif call_count == 57:
            return 8  # Drop below thermal_wait → trigger cooling
        elif call_count <= 65:
            return 8  # Cooling...
        else:
            return 25  # Cooled

    def mock_temp() -> int:
        return 60

    # Patch instance methods
    controller._get_gpu_tlimit = mock_tlimit  # type: ignore[method-assign]
    controller._get_gpu_temp = mock_temp  # type: ignore[method-assign]

    # Simulate thermal guard calls
    for _ in range(70):
        controller.throttle_guard()

    # After cycle with 55 iterations (> target_max=50), thermal_cont should decrease
    # Initial: 20, After: 20 - 3 = 17
    assert (
        controller.thermal_cont == 17
    ), f"Expected thermal_cont=17, got {controller.thermal_cont}"


def test_fixed_mode_no_adaptation() -> None:
    """Test that fixed mode does not adapt thermal_cont."""
    from nsight.thermovision import ThermalController

    # Create controller in fixed mode with custom values
    controller = ThermalController(
        thermal_mode="manual", thermal_wait=15, thermal_cont=30, verbose=False
    )

    assert controller.adaptive_mode is False
    assert controller.thermal_cont == 30
    assert controller.thermal_wait == 15

    call_count = 0

    def mock_tlimit() -> int:
        nonlocal call_count
        call_count += 1

        # Simulate few iterations (would trigger increase in adaptive mode)
        if call_count == 1:
            return 35  # Start counting
        elif call_count in [2, 3]:  # Only 2 iterations
            return 20
        elif call_count == 4:
            return 10  # Below thermal_wait → trigger cooling
        else:
            return 35  # Cooled

    def mock_temp() -> int:
        return 75

    # Patch instance methods
    controller._get_gpu_tlimit = mock_tlimit  # type: ignore[method-assign]
    controller._get_gpu_temp = mock_temp  # type: ignore[method-assign]

    # Simulate thermal guard calls
    for _ in range(10):
        controller.throttle_guard()

    # In fixed mode, thermal_cont should NOT change despite few iterations
    assert (
        controller.thermal_cont == 30
    ), f"Fixed mode should not adapt, but got {controller.thermal_cont}"
