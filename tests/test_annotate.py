# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
import torch

import nsight
from nsight import annotation
from nsight.utils import CUDA_CORE_AVAILABLE

configs = [(i,) for i in range(5)]


class DummyTestException(Exception):
    """Exception used for internal testing of API error handling."""

    pass


def test_annotate_context_manager_simple() -> None:
    @nsight.analyze.kernel(
        configs=configs, runs=7, verbosity=nsight.VerbosityLevel.INFO
    )
    def annotate_context_manager_simple(x: int) -> None:
        a = torch.randn(64, 64, device="cuda")
        b = torch.randn(64, 64, device="cuda")
        with nsight.annotate("test"):
            _ = a + b

    annotate_context_manager_simple()


def test_annotate_decorator_simple() -> None:
    @nsight.annotate("einsum")
    def einsum(a: torch.Tensor, b: torch.Tensor) -> Any:
        return a + b

    @nsight.analyze.kernel(
        configs=configs, runs=7, verbosity=nsight.VerbosityLevel.SILENT
    )
    def annotate_decorator_simple(n: int) -> None:
        a = torch.randn(64, 64, device="cuda")
        b = torch.randn(64, 64, device="cuda")

        einsum(a, b)

    annotate_decorator_simple()


@pytest.mark.parametrize(
    "ignore_failures",
    [
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                not CUDA_CORE_AVAILABLE,
                reason="cuda-core required for ignore_failures=True",
            ),
        ),
        False,
    ],
)  # type: ignore[untyped-decorator]
def test_annotate_context_manager(ignore_failures: bool) -> None:
    @nsight.analyze.kernel(
        configs=configs, runs=7, verbosity=nsight.VerbosityLevel.SILENT
    )
    def annotate_context_manager(n: int) -> None:
        a = torch.randn(64, 64, device="cuda")

        if ignore_failures == False:
            with pytest.raises(DummyTestException):
                with nsight.annotate(
                    "ignore_failures=false", ignore_failures=ignore_failures
                ):
                    # WAR until this issue is fixed: the parent process errors with ProfilerException if the ncu (child) process does not generate an ncu report. Hence adding a kernel launch before the exception to make sure an ncu report is generated
                    _ = a + a
                    raise DummyTestException()
        else:
            with nsight.annotate(
                "ignore_failures=true", ignore_failures=ignore_failures
            ):
                raise DummyTestException()

    annotate_context_manager()


@pytest.mark.parametrize(
    "ignore_failures",
    [
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                not CUDA_CORE_AVAILABLE,
                reason="cuda-core required for ignore_failures=True",
            ),
        ),
        False,
    ],
)  # type: ignore[untyped-decorator]
def test_annotate_decorator(ignore_failures: bool) -> None:
    @nsight.annotate(
        f"ignore_failures={ignore_failures}", ignore_failures=ignore_failures
    )
    def raise_exception(a: torch.Tensor, launch_kernel: bool) -> None:
        # WAR until this issue is fixed: the parent process errors with ProfilerException if the ncu (child) process does not generate an ncu report. Hence adding a kernel launch before the exception to make sure an ncu report is generated
        if launch_kernel:
            _ = a + a
        raise DummyTestException()

    @nsight.analyze.kernel(
        configs=configs, runs=7, verbosity=nsight.VerbosityLevel.SILENT
    )
    def annotate_decorator(n: int) -> None:
        a = torch.randn(64, 64, device="cuda")

        if ignore_failures == False:
            with pytest.raises(DummyTestException):
                raise_exception(a, True)

        else:
            raise_exception(a, False)

    annotate_decorator()


# ============================================================================
# Active annotations tracking tests
# ============================================================================


def test_active_annotations_nested() -> None:
    """Test that nested annotations raise ValueError with context manager."""

    @nsight.analyze.kernel(
        configs=[(64,)], runs=10, verbosity=nsight.VerbosityLevel.SILENT
    )
    def nested_annotation_test(n: int) -> None:
        a = torch.randn(n, n, device="cuda")
        b = torch.randn(n, n, device="cuda")

        # First annotation should work fine
        with nsight.annotate("outer_test"):
            _ = a + b

            # Trying to nest annotations should raise ValueError
            with pytest.raises(
                ValueError, match="Nested annotations are not supported"
            ):
                with nsight.annotate("inner_test"):
                    _ = a - b

    nested_annotation_test()


def test_active_annotations_duplicate_name() -> None:
    """Test that duplicate annotation names raise ValueError even when sequential."""

    @nsight.analyze.kernel(
        configs=[(64,)], runs=10, verbosity=nsight.VerbosityLevel.SILENT
    )
    def duplicate_annotation_test(n: int) -> None:
        a = torch.randn(n, n, device="cuda")
        b = torch.randn(n, n, device="cuda")

        # First annotation should work fine
        with nsight.annotate("duplicate_test"):
            _ = a + b

        # Trying to use the same name again (even sequentially) should raise ValueError
        with pytest.raises(
            ValueError, match="Annotation name 'duplicate_test' has already been used"
        ):
            with nsight.annotate("duplicate_test"):
                _ = a - b

    duplicate_annotation_test()


def test_active_annotations_duplicate_name_decorator() -> None:
    """Test that nested annotations raise ValueError with decorator."""

    @nsight.annotate("nested_decorator_test")
    def annotated_function(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b

    @nsight.analyze.kernel(
        configs=[(64,)], runs=10, verbosity=nsight.VerbosityLevel.SILENT
    )
    def duplicate_annotation_decorator_test(n: int) -> None:
        a = torch.randn(n, n, device="cuda")
        b = torch.randn(n, n, device="cuda")

        # Using a context manager and then calling a decorated function should fail
        with nsight.annotate("outer_annotation"):
            _ = a + b

            with pytest.raises(
                ValueError,
                match="Nested annotations are not supported",
            ):
                _ = annotated_function(a, b)

    duplicate_annotation_decorator_test()
