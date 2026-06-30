# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
This is used to test the scenario when a user initializes CUDA before importing nsight.
NCU attach-mode requires CUDA to not be initialized before the injection library is loaded
so profiling fails with an error like "Cuda is initialized before the tool".
"""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

torch.cuda.init()  # There can be any other torch call that initializes CUDA before nsight is imported

import nsight


@nsight.analyze.kernel()
def run_simple_kernel(n: int) -> None:
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")
    with nsight.annotate("test"):
        _ = a @ b


def test() -> None:
    """Run this program as a subprocess; profiling must fail with the expected error."""
    this_file = Path(__file__).resolve()
    result = subprocess.run(
        [sys.executable, str(this_file)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=this_file.parent.parent,
    )

    # Process should fail with a ProfilerException raised by begin_profiling()
    assert result.returncode != 0, "Subprocess unexpectedly succeeded"

    stderr_lower = (result.stderr or "").lower()
    # Expect the standardized injection error message with the code emitted by ncu injection
    # When CUDA is already initialized before injection, we expect code 3 (NV_INJ_ERROR_NOT_INITIALIZED)
    assert (
        "failed to begin profiling" in stderr_lower
    ), f"Unexpected stderr: {result.stderr}"
    assert (
        "(code=3)" in stderr_lower
    ), f"Expected injection error code 3 in stderr, got: {result.stderr}"


if __name__ == "__main__":
    run_simple_kernel(64)
