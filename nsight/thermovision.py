# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from typing import Any, Literal

from nsight.exceptions import CoolingTimeoutError

"""
This module provides GPU thermal monitoring and throttling prevention using NVIDIA's NVML library
(as exposed through cuda.core.system).

It monitors GPU temperature and T.limit, and delays execution when the GPU
is too hot to avoid thermal throttling. The module uses an adaptive approach that
learns optimal cooling thresholds based on workload characteristics.
"""

# Guard NVML imports
try:
    from cuda.core import Device as CudaDevice
    from cuda.core import system

    CUDA_CORE_AVAILABLE = True
except ImportError:
    CUDA_CORE_AVAILABLE = False
    print(
        "Warning: Cannot import cuda.core. Ensure nsight-python was installed properly with all dependencies."
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
        thermal_device: int | None = None,
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
            thermal_device: CUDA device ordinal to monitor for thermal throttling.
                If set, monitor that CUDA device. If None, monitor the current
                CUDA device context. Device numbering follows CUDA runtime ordinals and
                honors ``CUDA_VISIBLE_DEVICES``.
                Default: None
            verbose: Whether to print thermal messages.
                Default: False
        """
        self.device: Any = None
        self.thermal_device = thermal_device
        self._current_cuda_device_id: int | None = None
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

        Resolves the NVML device that corresponds to the profiled CUDA device
        (honoring ``CUDA_VISIBLE_DEVICES``).

        Returns:
            True if temperature retrieval is supported, False otherwise.
        """
        if not CUDA_CORE_AVAILABLE:
            return False

        if self.device is None:
            resolved_device = self._resolve_device()
            if resolved_device is None:
                return False
            self.device = resolved_device

        return self._is_temp_retrieval_supported()

    def _map_cuda_to_system_device(self, cuda_ordinal: int | None) -> Any:
        """Map a CUDA device ordinal to its corresponding NVML system device.

        Args:
            cuda_ordinal: CUDA ordinal to resolve. ``None`` means current CUDA
                device context.

        Returns:
            The corresponding NVML ``system.Device``.
        """
        cuda_device = CudaDevice(cuda_ordinal)
        system_device = cuda_device.to_system_device()
        self._current_cuda_device_id = cuda_device.device_id
        return system_device

    def _resolve_device(self) -> Any:
        """Resolve the NVML device to monitor.

        The CUDA device (explicitly requested via ``thermal_device`` or the
        current CUDA device context) is mapped to its underlying physical
        NVML device by UUID. This ensures the GPU actually running the profiled
        kernels is monitored, even when ``CUDA_VISIBLE_DEVICES`` remaps device
        ordinals.

        Returns:
            The resolved NVML ``system.Device`` to monitor, or ``None`` if no
            device could be resolved.
        """
        cuda_ordinal = self.thermal_device
        try:
            # ``None`` selects the current CUDA device and already reflects
            # CUDA_VISIBLE_DEVICES. Mapping to system device is done by UUID.
            system_device = self._map_cuda_to_system_device(cuda_ordinal)
            if self.verbose:
                print(
                    f"[Thermovision] Monitoring CUDA device {self._current_cuda_device_id} "
                    f"(NVML index {system_device.index}, {system_device.name}, UUID {system_device.uuid})"
                )
            return system_device
        except Exception as e:
            # Fall back to physical GPU 0 so thermal protection still works on
            # single-GPU systems and when CUDA cannot be initialized here.
            if cuda_ordinal is not None:
                print(
                    f"Warning: Could not resolve thermal_device={cuda_ordinal} "
                    f"to an NVML device ({e}). Falling back to physical GPU 0."
                )
            self._current_cuda_device_id = None
            try:
                return system.Device(index=0)
            except Exception:
                return None

    def _refresh_device_for_current_cuda_context(self) -> None:
        """Refresh monitored NVML device when current CUDA device changes.

        This applies only when ``thermal_device`` is not explicitly set. It keeps
        Thermovision aligned with runtime CUDA context switches in user code
        (for example via ``torch.cuda.set_device``).
        """
        if self.thermal_device is not None:
            return

        try:
            current_cuda_id = CudaDevice(None).device_id
            if (
                self.device is not None
                and self._current_cuda_device_id == current_cuda_id
            ):
                return

            self.device = self._map_cuda_to_system_device(None)
            if self.verbose:
                print(
                    f"[Thermovision] Switched monitoring to CUDA device {self._current_cuda_device_id} "
                    f"(NVML index {self.device.index}, {self.device.name}, UUID {self.device.uuid})"
                )
        except Exception:
            # Keep existing device if refresh fails. If no device exists yet,
            # try the regular resolution path (including fallback).
            if self.device is None:
                self.device = self._resolve_device()

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
        self._refresh_device_for_current_cuda_context()
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
            return int(self.device.temperature.margin)
        except system.NotSupportedError as e:
            print("Error: GPU does not support temperature limit retrieval:", e)
            return None
        except Exception as e:
            raise e

    def _get_gpu_temp(self) -> int:
        """Get current GPU temperature.

        Returns:
            GPU temperature in degrees Celsius
        """
        return int(self.device.temperature.get_sensor())

    def _is_temp_retrieval_supported(self) -> bool:
        """Check if GPU supports temperature retrieval.

        Returns:
            True if supported, False otherwise
        """
        try:
            self.device.temperature.margin
            return True
        except Exception:
            print(
                "Warning: Nsight Python Thermovision is not supported on this machine"
            )
            return False
