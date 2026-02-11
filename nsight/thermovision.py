# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from typing import Any, Literal

from nsight.exceptions import CoolingTimeoutError

"""
This module provides GPU thermal monitoring and throttling prevention using NVIDIA's NVML library.

It monitors GPU temperature and T.limit, and delays execution when the GPU
is too hot to avoid thermal throttling. The module uses an adaptive approach that
learns optimal cooling thresholds based on workload characteristics.
"""

# Guard NVML imports
try:
    from pynvml import (
        NVML_TEMPERATURE_GPU,
        NVMLError_NotSupported,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMarginTemperature,
        nvmlDeviceGetTemperature,
        nvmlInit,
    )

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print(
        "Warning: Cannot import pynvml (provided by nvidia-ml-py). Ensure nsight-python was installed properly with all dependencies."
    )

# Default thermal threshold constants
DEFAULT_THERMAL_WAIT: int = 10  # Default wait threshold (trigger cooling)
DEFAULT_THERMAL_CONT: int = 20  # Default continue threshold (resume after cooling)
DEFAULT_THERMAL_TIMEOUT: int = 180  # Default timeout in seconds (wait for cooling)

# Adaptive control constants
MIN_THERMAL_CONT: int = 12  # Minimum continue threshold in °C (safety floor)
MAX_THERMAL_CONT: int = 50  # Maximum continue threshold in °C (prevents over-cooling)
TARGET_MIN_ITERATIONS: int = 5  # If < 5 iterations, GPU heats too quickly → cool more
TARGET_MAX_ITERATIONS: int = 50  # If > 50 iterations, GPU heats slowly → cool less
ADJUSTMENT_STEP: int = 3  # Adjust thermal_cont by 3°C per adaptation step


class ThermalController:
    """GPU thermal monitoring and throttling prevention.

    Manages GPU temperature and prevents thermal throttling by pausing profiling
    when the GPU gets too hot and resuming after cooling.
    """

    def __init__(
        self,
        thermal_mode: Literal["auto", "manual", "off"] = "auto",
        thermal_wait: int | None = None,
        thermal_cont: int | None = None,
        thermal_timeout: int | None = None,
        verbose: bool = False,
    ):
        """Initialize thermal controller.

        Args:
            thermal_mode: Thermal control mode.
                - "auto": Adaptive mode - automatically adjusts thermal_cont based on GPU behavior.
                - "manual": Fixed mode - uses specified thresholds without adaptation.
                - "off": Thermal control disabled.
                Default: "auto"
            thermal_wait: Wait threshold (thermal headroom in °C to trigger cooling).
                If None, uses default (10°C).
                Default: None
            thermal_cont: Continue threshold (thermal headroom in °C to resume after cooling).
                If None, uses default (20°C).
                Default: None
            thermal_timeout: Maximum wait time in seconds for GPU to cool down.
                If None, uses default (180 seconds).
                Default: None
            verbose: Whether to print thermal messages.
                Default: False
        """
        self.handle: Any = None
        self.thermal_mode = thermal_mode

        # Set adaptive_mode based on thermal_mode
        self.adaptive_mode = thermal_mode == "auto"

        # Set thresholds (use defaults if None)
        self.thermal_wait = (
            thermal_wait if thermal_wait is not None else DEFAULT_THERMAL_WAIT
        )
        self.thermal_cont = (
            thermal_cont if thermal_cont is not None else DEFAULT_THERMAL_CONT
        )
        self.thermal_timeout = (
            thermal_timeout if thermal_timeout is not None else DEFAULT_THERMAL_TIMEOUT
        )

        # Configuration
        self.verbose = verbose

        # Runtime state (tracks current thermal behavior)
        self.iterations_since_cont = 0
        self.is_counting = False

    def init(self) -> bool:
        """Initialize NVML and get GPU handle.

        Returns:
            True if temperature retrieval is supported, False otherwise.
        """
        if not PYNVML_AVAILABLE:
            return False

        if self.handle is None:
            nvmlInit()
            self.handle = nvmlDeviceGetHandleByIndex(0)

        return self._is_temp_retrieval_supported()

    def throttle_guard(self) -> None:
        """Check thermal state and pause if GPU is too hot.

        Thermal headroom = temperature margin before GPU starts throttling.

        Operates in two modes:
        - Auto mode: Automatically adjusts thermal_cont based on workload
        - Manual mode: Uses user-provided thresholds without adaptation

        Adaptive Algorithm (in auto mode):
        1. When thermal headroom reaches thermal_cont after cooling, start counting iterations
        2. Run kernel as headroom drops from thermal_cont toward thermal_wait
        3. When headroom drops below thermal_wait, analyze iteration count:
           - Few iterations (<TARGET_MIN_ITERATIONS): GPU heats quickly → increase thermal_cont (cool more)
           - Many iterations (>TARGET_MAX_ITERATIONS): GPU heats slowly → decrease thermal_cont (cool less)
        4. Wait until GPU cools back to thermal_cont, then repeat
        """
        tlimit = self._get_gpu_tlimit()
        if tlimit is None:
            return

        # Increment counter FIRST - count the run that's about to execute
        if self.is_counting:
            self.iterations_since_cont += 1

        # Thermal pause trigger - when GPU gets too hot
        if tlimit <= self.thermal_wait:
            self._handle_thermal_pause(tlimit)
            return

        # Start counting trigger - when GPU has cooled enough
        if tlimit >= self.thermal_cont and not self.is_counting:
            self.is_counting = True

    def _handle_thermal_pause(self, tlimit: int) -> None:
        """Handle thermal pause and cooling.

        Args:
            tlimit: Current thermal headroom value
        """
        temperature = self._get_gpu_temp()
        if self.verbose:
            print(
                f"\n[Thermovision] GPU hot: {temperature}°C (thermal headroom: {tlimit}°C)"
            )

        # Adaptive adjustment
        if self.adaptive_mode and self.is_counting and self.iterations_since_cont > 0:
            if self.iterations_since_cont < TARGET_MIN_ITERATIONS:
                self.thermal_cont = min(
                    self.thermal_cont + ADJUSTMENT_STEP, MAX_THERMAL_CONT
                )
            elif self.iterations_since_cont > TARGET_MAX_ITERATIONS:
                self.thermal_cont = max(
                    self.thermal_cont - ADJUSTMENT_STEP, MIN_THERMAL_CONT
                )

        self.iterations_since_cont = 0
        self.is_counting = False

        # Wait for GPU to cool down
        self._wait_for_cooling(tlimit)

        # Start counting immediately after cooling completes
        self.is_counting = True

    def _wait_for_cooling(self, initial_tlimit: int) -> None:
        """Wait for GPU to cool to target threshold.

        Args:
            initial_tlimit: Initial thermal headroom value when cooling started
        """
        start_time = time.time()
        tlimit: int | None = initial_tlimit

        # Loop until GPU cools to thermal_cont threshold or timeout occurs
        # Default timeout: 180 seconds (thermal_timeout parameter)
        while tlimit is not None and tlimit < self.thermal_cont:
            elapsed = time.time() - start_time
            if elapsed > self.thermal_timeout:
                self._handle_cooling_timeout(tlimit)
                start_time = time.time()  # Reset timeout

            # Poll every 0.5 seconds
            # 0.5 seconds is fast enough to detect when cooling target is achieved
            # without excessive CPU usage from constant polling
            time.sleep(0.5)
            tlimit = self._get_gpu_tlimit()

        # Calculate wait time and final temperature
        wait_time = time.time() - start_time
        temperature = self._get_gpu_temp()
        if self.verbose:
            print(
                f"[Thermovision] Waited {wait_time:.1f}s → Cooled to {temperature}°C (thermal headroom: {tlimit}°C)\n"
            )

    def _handle_cooling_timeout(self, tlimit: int) -> None:
        """Handle cooling timeout.

        Args:
            tlimit: Current thermal headroom value

        Raises:
            CoolingTimeoutError: If cooling cannot reach target threshold
        """
        if self.adaptive_mode:
            if self.verbose:
                print(
                    f"## Adaptive mode: decreasing thermal_cont to {self.thermal_cont - ADJUSTMENT_STEP}°C"
                )
            self.thermal_cont = max(
                self.thermal_cont - ADJUSTMENT_STEP, MIN_THERMAL_CONT
            )
            if self.thermal_cont == MIN_THERMAL_CONT and self.thermal_cont > tlimit:
                raise CoolingTimeoutError(
                    f"Adaptive thermal control reached minimum thermal_cont={MIN_THERMAL_CONT}°C "
                    f"but GPU thermal headroom is only {tlimit}°C. Cannot continue profiling."
                )
        else:
            raise CoolingTimeoutError(
                f"Timeout after {self.thermal_timeout}s. Cannot reach thermal_cont={self.thermal_cont}°C "
                f"(current thermal headroom: {tlimit}°C). "
                f"Try: increase thermal_wait, decrease thermal_cont, or increase thermal_timeout."
            )

    def _get_gpu_tlimit(self) -> int | None:
        """Get GPU thermal limit (T.Limit).

        Returns:
            Thermal headroom in degrees Celsius, or None if not supported
        """
        try:
            return nvmlDeviceGetMarginTemperature(self.handle)  # type: ignore[no-any-return]
        except NVMLError_NotSupported as e:
            print("Error: GPU does not support temperature limit retrieval:", e)
            return None
        except Exception as e:
            raise e

    def _get_gpu_temp(self) -> int:
        """Get current GPU temperature.

        Returns:
            GPU temperature in degrees Celsius
        """
        return nvmlDeviceGetTemperature(self.handle, NVML_TEMPERATURE_GPU)  # type: ignore[no-any-return]

    def _is_temp_retrieval_supported(self) -> bool:
        """Check if GPU supports temperature retrieval.

        Returns:
            True if supported, False otherwise
        """
        try:
            nvmlDeviceGetMarginTemperature(self.handle)
            return True
        except Exception:
            print(
                "Warning: Nsight Python Thermovision is not supported on this machine"
            )
            return False
