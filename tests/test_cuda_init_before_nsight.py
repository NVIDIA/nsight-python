# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
This is used to test the scenario when a user initializes CUDA before importing nsight.
NCU attach-mode requires CUDA to not be initialized before the injection library is loaded
so profiling fails with an error like "Cuda is initialized before the tool".

The test runs in a fresh subprocess via pytest-isolated, because nsight must not have
been imported yet. The root conftest.py skips its tool setup in that subprocess.
"""

import pytest
import torch


@pytest.mark.isolated  # type: ignore[untyped-decorator]
def test_cuda_init_before_nsight() -> None:
    """Profiling must fail with the expected error when CUDA is initialized first."""
    torch.cuda.init()  # any torch call that initializes CUDA would do

    # Importing nsight (or, with NSPY_NCU_INIT_AT_IMPORT=0, the first decorated call)
    # loads NCU, and the test needs that to happen after CUDA init.
    with pytest.raises(Exception) as exc_info:
        import nsight

        @nsight.analyze.kernel()
        def run_simple_kernel(n: int) -> None:
            a = torch.randn(n, n, device="cuda")
            b = torch.randn(n, n, device="cuda")
            with nsight.annotate("test"):
                _ = a @ b

        run_simple_kernel(64)

    from nsight.exceptions import ProfilerException

    assert isinstance(
        exc_info.value, ProfilerException
    ), f"Unexpected exception: {exc_info.value!r}"
    message = str(exc_info.value).lower()
    # Expect the standardized injection error message with the code emitted by ncu injection.
    # When CUDA is already initialized before injection, we expect code 3
    # (NV_INJ_ERROR_NOT_INITIALIZED)
    assert "failed to begin profiling" in message, f"Unexpected error: {message}"
    assert "(code=3)" in message, f"Expected injection error code 3, got: {message}"
