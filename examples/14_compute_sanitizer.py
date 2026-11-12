# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal Compute Sanitizer example with an out-of-bounds device write.

Activation self-injects the sanitizer backend, so this script and its CUDA
workload each execute once.
"""

# Disable legacy import-time NCU activation before importing nsight so Compute
# Sanitizer can be explicitly selected below.
import os

os.environ["NSPY_NCU_INIT_AT_IMPORT"] = "0"

import nsight


def main() -> None:
    # Load the backend before importing CUDA modules or constructing a device.
    nsight.activate(
        nsight.Tool.COMPUTE_SANITIZER,
        options=nsight.ComputeSanitizerOptions(
            tool=nsight.ComputeSanitizerTool.MEMCHECK
        ),
    )

    from cuda.bindings import driver
    from cuda.core import Device

    device = Device(0)
    device.set_current()
    allocation_stream = device.create_stream()

    def run_invalid_memset(allocation_size: int) -> None:
        allocation = device.allocate(allocation_size, stream=allocation_stream)
        allocation_stream.sync()

        with nsight.annotate("out-of-bounds-memset"):
            (status,) = driver.cuMemsetD8(allocation.handle, 0, allocation_size + 4)

        if status != driver.CUresult.CUDA_ERROR_INVALID_VALUE:
            raise RuntimeError(f"Unexpected cuMemsetD8 status: {status}")

    run_invalid_memset(4)
    print(f"Compute Sanitizer log: {nsight.sanitize.log_path()}")


if __name__ == "__main__":
    main()
