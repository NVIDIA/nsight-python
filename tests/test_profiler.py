# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test suite for Nsight Python profiler functionality.
"""

import os
import shutil
from collections.abc import Generator, Sequence
from typing import Any, Literal

import pytest
import torch
from cuda.core.experimental import Device, LaunchConfig, Program, launch

import nsight
from nsight import exceptions

# Common CUDA kernel code for tests that launch multiple kernels
CUDA_KERNEL_CODE = """
extern "C" __global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

extern "C" __global__ void vector_multiply(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}
"""


def _simple_kernel_impl(x: int, y: int, annotation: str = "test") -> None:
    """Shared kernel implementation for testing."""
    a = torch.randn(x, y, device="cuda")
    b = torch.randn(x, y, device="cuda")
    with nsight.annotate(annotation):
        _ = a + b


# ============================================================================
# Decorator syntax tests
# ============================================================================


@nsight.analyze.kernel
def decorator_without_parens(x: int, y: int) -> None:
    """Test using @nsight.analyze.kernel without parentheses."""
    _simple_kernel_impl(x, y)


def test_decorator_without_parens() -> None:
    """Test that decorator works without parentheses."""
    decorator_without_parens(42, 23)


# ----------------------------------------------------------------------------


@nsight.analyze.kernel()
def decorator_with_empty_parens(x: int, y: int) -> None:
    """Test using @nsight.analyze.kernel() with empty parentheses."""
    _simple_kernel_impl(x, y)


def test_decorator_with_empty_parens() -> None:
    """Test that decorator works with empty parentheses."""
    decorator_with_empty_parens(42, 23)


# ----------------------------------------------------------------------------


@nsight.analyze.kernel(runs=3)
def decorator_with_args(x: int, y: int) -> None:
    """Test using @nsight.analyze.kernel(args) with arguments."""
    _simple_kernel_impl(x, y)


def test_decorator_with_args() -> None:
    """Test that decorator works with arguments."""
    decorator_with_args(42, 23)


# ----------------------------------------------------------------------------


@nsight.analyze.kernel
def decorator_without_parens_with_configs(x: int, y: int) -> None:
    """Test using @nsight.analyze.kernel without parentheses, configs at call time."""
    _simple_kernel_impl(x, y)


def test_decorator_without_parens_with_configs() -> None:
    """Test that decorator without parens works with configs at call time."""
    decorator_without_parens_with_configs(configs=[(42, 23), (12, 13)])


# ============================================================================
# Configuration handling tests
# ============================================================================


@nsight.analyze.kernel()
def config_at_call_time_positional(x: int, y: int) -> None:
    _simple_kernel_impl(x, y)


def test_config_at_call_time_positional() -> None:
    """Test providing configuration as positional arguments at call time."""
    config_at_call_time_positional(42, 23)


# ----------------------------------------------------------------------------


@nsight.analyze.kernel()
def config_at_call_time_configs(x: int, y: int) -> None:
    _simple_kernel_impl(x, y)


def test_config_at_call_time_configs() -> None:
    """Test providing configuration as configs list at call time."""
    config_at_call_time_configs(configs=[(42, 23), (12, 13)])


# ----------------------------------------------------------------------------


def test_config_at_call_time_with_kwargs() -> None:
    """Test that keyword arguments raise appropriate error."""
    with pytest.raises(
        exceptions.ProfilerException, match="Keyword arguments are not supported yet"
    ):
        config_at_call_time_configs(42, y=23)


# ----------------------------------------------------------------------------


def test_config_at_call_time_repeated() -> None:
    """Test providing configuration(s) to multiple calls"""
    result_a = config_at_call_time_configs(42, 23)
    result_b = config_at_call_time_configs(configs=[(17, 92), (18, 93)])

    assert len(result_a.to_dataframe()) == 1
    assert len(result_b.to_dataframe()) == 2


# ----------------------------------------------------------------------------
# Tests for functions with no arguments
# ----------------------------------------------------------------------------


@nsight.analyze.kernel
def no_args_function_no_parens() -> None:
    """Test function with no arguments using decorator without parentheses."""
    a = torch.randn(64, 64, device="cuda")
    b = torch.randn(64, 64, device="cuda")
    with nsight.annotate("test"):
        _ = a + b


def test_no_args_function_no_parens() -> None:
    """Test that function with no args works without providing configs."""
    # Should work without configs since the function takes no arguments
    # and we just want to run it once (or with default runs).
    no_args_function_no_parens()


# ----------------------------------------------------------------------------


@nsight.analyze.kernel()
def no_args_function_with_parens() -> None:
    """Test function with no arguments using decorator with empty parentheses."""
    a = torch.randn(64, 64, device="cuda")
    b = torch.randn(64, 64, device="cuda")
    with nsight.annotate("test"):
        _ = a + b


def test_no_args_function_with_parens() -> None:
    """Test that function with no args works with empty parentheses."""
    # Should work without configs since the function takes no arguments
    # and we just want to run it once (or with default runs).
    result = no_args_function_with_parens()

    # Verify the dataframe structure
    assert result is not None, "ProfileResults should be returned"
    df = result.to_dataframe()

    # Should have exactly 1 row (1 annotation, 1 config with no params)
    assert len(df) == 1, f"Expected 1 row in dataframe, got {len(df)}"

    # Verify annotation name
    assert df["Annotation"].iloc[0] == "test", "Expected annotation 'test'"

    # Verify metric value is reasonable (should be positive)
    assert df["AvgValue"].iloc[0] > 0, "Expected positive metric value"


# ----------------------------------------------------------------------------


@nsight.analyze.kernel(runs=3)
def no_args_function_with_kwargs() -> None:
    """Test function with no arguments using decorator with keyword arguments."""
    a = torch.randn(64, 64, device="cuda")
    b = torch.randn(64, 64, device="cuda")
    with nsight.annotate("test"):
        _ = a + b


def test_no_args_function_with_kwargs() -> None:
    """Test that function with no args works when decorator has kwargs."""
    # Should work without configs since the function takes no arguments
    # and we just want to run it multiple times with the specified runs.
    result = no_args_function_with_kwargs()

    # Verify the dataframe structure
    assert result is not None, "ProfileResults should be returned"
    df = result.to_dataframe()

    # Should have exactly 1 row (1 annotation, 1 config with no params)
    assert len(df) == 1, f"Expected 1 row in dataframe, got {len(df)}"

    # Verify that runs=3 was respected
    assert df["NumRuns"].iloc[0] == 3, f"Expected 3 runs, got {df['NumRuns'].iloc[0]}"


# ----------------------------------------------------------------------------


def test_no_args_vs_with_args_dataframe_comparison() -> None:
    """Compare dataframe structure for functions with and without arguments."""

    # Test function with no args
    @nsight.analyze.kernel(output="quiet")
    def no_args() -> None:
        a = torch.randn(64, 64, device="cuda")
        b = torch.randn(64, 64, device="cuda")
        with nsight.annotate("test"):
            _ = a + b

    # Test function with args
    @nsight.analyze.kernel(configs=[(32,), (64,)], output="quiet")
    def with_args(size: int) -> None:
        a = torch.randn(size, size, device="cuda")
        b = torch.randn(size, size, device="cuda")
        with nsight.annotate("test"):
            _ = a + b

    result_no_args = no_args()
    result_with_args = with_args()

    assert result_no_args is not None
    assert result_with_args is not None

    df_no_args = result_no_args.to_dataframe()
    df_with_args = result_with_args.to_dataframe()

    # No-args function should have 1 row (1 config)
    assert (
        len(df_no_args) == 1
    ), f"No-args function should have 1 row, got {len(df_no_args)}"

    # With-args function should have 2 rows (2 configs)
    assert (
        len(df_with_args) == 2
    ), f"With-args function should have 2 rows, got {len(df_with_args)}"

    assert (
        "size" in df_with_args.columns
    ), "With-args function should have 'size' column"

    # Verify the size values in the dataframe match the configs
    assert set(df_with_args["size"].values) == {32, 64}


# ----------------------------------------------------------------------------


def test_no_args_function_with_derive_metric() -> None:
    """Test that derive_metric works with functions that have no arguments."""

    # Define a derive_metric function that only takes the metric values
    # (no config parameters since the function has no args)
    def custom_metric(time_ns: float) -> float:
        """Convert time to arbitrary custom metric."""
        return time_ns / 1e6  # Convert to milliseconds

    @nsight.analyze.kernel(runs=2, output="quiet", derive_metric=custom_metric)
    def no_args_with_transform() -> None:
        a = torch.randn(128, 128, device="cuda")
        b = torch.randn(128, 128, device="cuda")
        with nsight.annotate("test"):
            _ = a + b

    result = no_args_with_transform()

    assert result is not None, "ProfileResults should be returned"
    df = result.to_dataframe()

    # Should have exactly 1 row
    assert len(df) == 1, f"Expected 1 row in dataframe, got {len(df)}"

    # Verify the transformation was applied
    assert (
        df["Transformed"].iloc[0] == "custom_metric"
    ), f"Expected 'custom_metric' in Transformed column, got {df['Transformed'].iloc[0]}"

    # Verify the value is positive (transformed metric should still be positive)
    assert df["AvgValue"].iloc[0] > 0, "Expected positive transformed metric value"

    # Verify runs parameter was respected
    assert df["NumRuns"].iloc[0] == 2, f"Expected 2 runs, got {df['NumRuns'].iloc[0]}"


# ----------------------------------------------------------------------------


@pytest.mark.parametrize("config_type", ["decoration", "call_time"])  # type: ignore[untyped-decorator]
def test_config_non_sized_iterables(config_type: str) -> None:
    """Test for config iterables which do not implement __len__()"""

    def generate_config() -> Generator[tuple[int, int]]:
        for i in range(1, 6):
            yield i, i + 1

    decorator_configs = generate_config() if config_type == "decoration" else None
    call_time_configs = generate_config() if config_type == "call_time" else None

    @nsight.analyze.kernel(configs=decorator_configs)
    def non_sized_iterables(x: int, y: int) -> None:
        _simple_kernel_impl(x, y, annotation="non_sized_iterables")

    non_sized_iterables(configs=call_time_configs)


# ----------------------------------------------------------------------------


def generate_config() -> Generator[int]:
    for i in range(1, 6):
        yield i


@pytest.mark.parametrize(
    "configs, size_of_each_config",
    [
        ([(100,), 200, (300,)], 1),
        (generate_config(), None),
        (range(1, 4), None),
        ([(100, 200), 200, (300, 400)], 2),
    ],
)  # type: ignore[untyped-decorator]
def test_configs_with_scalar_values(
    configs: list[list[int] | int] | Generator[int] | range,
    size_of_each_config: int | None,
) -> None:

    @nsight.analyze.kernel(configs=configs)
    def configs_with_scalar_values(n: int) -> None:
        _simple_kernel_impl(n, n)

    if size_of_each_config == 1:
        result = configs_with_scalar_values()
        df = result.to_dataframe()
        # Check if the configs are correct
        assert df["n"].to_list() == [100, 200, 300]

    elif size_of_each_config == 2:
        with pytest.raises(
            exceptions.ProfilerException,
            match="All configs must have the same number of arguments. Found lengths",
        ):
            configs_with_scalar_values()
    else:
        if isinstance(configs, Generator):
            # Define a different function to avoid issues arising from script relaunch
            @nsight.analyze.kernel(configs=configs)
            def configs_with_scalar_values_generators(n: int) -> None:
                _simple_kernel_impl(n, n)

            result = configs_with_scalar_values_generators()
            df = result.to_dataframe()

            # Check if the configs are correct
            assert df["n"].to_list() == [1, 2, 3, 4, 5]

        elif isinstance(configs, range):
            # Define a different function to avoid issues arising from script relaunch
            @nsight.analyze.kernel(configs=configs)
            def configs_with_scalar_values_range(n: int) -> None:
                _simple_kernel_impl(n, n)

            result = configs_with_scalar_values_range()
            df = result.to_dataframe()

            # Check if the configs are correct
            assert df["n"].to_list() == [1, 2, 3]


# ----------------------------------------------------------------------------


# ============================================================================
# Parameter validation tests
# ============================================================================


@nsight.analyze.kernel(configs=[(1,), (2,)])
def function_with_default_parameter(x: int, y: Any = None) -> None:
    a = torch.randn(x, x, device="cuda")
    b = torch.randn(x, x, device="cuda")
    with nsight.annotate("test"):
        _ = a + b


def test_function_with_default_parameter() -> None:
    """Test that calling function with defaults without providing all args raises error."""
    with pytest.raises(exceptions.ProfilerException):
        function_with_default_parameter()


# ============================================================================
# Kernel execution tests
# ============================================================================

configs = [(i,) for i in range(5)]


@nsight.analyze.plot()
@nsight.analyze.kernel(configs=configs, runs=7, output="verbose")
def simple(x: int) -> None:
    a = torch.randn(64, 64, device="cuda")
    b = torch.randn(64, 64, device="cuda")
    with nsight.annotate("test"):
        _ = a + b


def test_simple() -> None:
    """Test basic kernel execution with multiple configurations."""
    simple()


# ----------------------------------------------------------------------------
# Conditional execution tests
# ----------------------------------------------------------------------------


@nsight.analyze.plot()
@nsight.analyze.kernel(configs=configs, runs=7, output="quiet")
def different_kernels(x: int) -> None:
    a = torch.randn(64, 64, device="cuda")
    b = torch.randn(64, 64, device="cuda")
    with nsight.annotate("test"):
        if x % 2 == 0:
            _ = a - b
        else:
            _ = a + b


def test_different_kernels() -> None:
    """Test kernel with conditional execution paths."""
    different_kernels()


# ----------------------------------------------------------------------------
# Multiple kernels per run tests
# ----------------------------------------------------------------------------


@nsight.analyze.plot()
@nsight.analyze.kernel(
    configs=configs,
    runs=7,
    combine_kernel_metrics=lambda x, y: x + y,
)
def multiple_kernels_per_run(x: int) -> None:
    a = torch.randn(64, 64, device="cuda")
    b = torch.randn(64, 64, device="cuda")
    with nsight.annotate("test"):
        _ = a @ b
        _ = a @ b
        _ = a @ b


def test_multiple_kernels_per_run() -> None:
    """Test kernel that launches multiple operations per execution."""
    multiple_kernels_per_run()


# ----------------------------------------------------------------------------
# Varying kernels per run tests (currently unsupported)
# ----------------------------------------------------------------------------


@nsight.analyze.plot()
@nsight.analyze.kernel(
    configs=((False,), (True,)),
    runs=3,
    combine_kernel_metrics=lambda x, y: x + y,
)
def varying_multiple_kernels_per_run(flag: bool) -> None:
    a = torch.randn(64, 64, device="cuda")
    b = torch.randn(64, 64, device="cuda")
    with nsight.annotate("test"):
        _ = a + b
        if flag:
            _ = a + b


@pytest.mark.skip("don't yet support varying number of kernels per run")  # type: ignore[untyped-decorator]
def test_varying_multiple_kernels_per_run() -> None:
    """Test kernel with varying number of operations per run (currently unsupported)."""
    varying_multiple_kernels_per_run()


@nsight.analyze.kernel(
    configs=(
        (1,),
        (2,),
        (3,),
    ),
    runs=3,
    normalize_against="annotation1",
)
def normalize_against(n: int) -> None:
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")
    with nsight.annotate("annotation1"):
        _ = a + b

    with nsight.annotate("annotation2"):
        _ = a - b


def test_parameter_normalize_against() -> None:
    profile_output = normalize_against()
    if profile_output is not None:
        df = profile_output.to_dataframe()

        # Basic validation for normalize_against: AvgValue for the annotation being used as normalization factor should be 1
        assert (df.loc[df["Annotation"] == "annotation1", "AvgValue"] == 1).all()


@nsight.analyze.kernel(
    configs=(
        (1,),
        (2,),
        (3,),
    ),
    runs=3,
    normalize_against="annotation1",
    # Some parameters that have a numerical determinism greater than
    # 1 and grow with substantial increases in n.
    metrics=[
        "smsp__inst_executed.sum",
        "smsp__inst_issued.sum",
        "smsp__sass_inst_executed_op_global_ld.sum",
        "smsp__sass_inst_executed_op_global_st.sum",
    ],
)
def normalize_against_multiple_metrics(n: int) -> None:
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")

    c = torch.randn(100 * n, 100 * n, device="cuda")
    d = torch.randn(100 * n, 100 * n, device="cuda")

    with nsight.annotate("annotation1"):
        _ = a + b

    with nsight.annotate("annotation2"):
        _ = c + d


@pytest.mark.xfail(  # type: ignore[untyped-decorator]
    reason="Waiting for proper support for standard normalization and speedup computation"
)
def test_parameter_normalize_against_multiple_metrics() -> None:
    profile_output = normalize_against_multiple_metrics()
    if profile_output is not None:
        df = profile_output.to_dataframe()

        requested_metrics = [
            "smsp__inst_executed.sum",
            "smsp__inst_issued.sum",
            "smsp__sass_inst_executed_op_global_ld.sum",
            "smsp__sass_inst_executed_op_global_st.sum",
        ]

        for annotation in ["annotation1", "annotation2"]:
            for n in [1, 2, 3]:
                subset = df[(df["Annotation"] == annotation) & (df["n"] == n)]
                assert len(subset) == len(requested_metrics)

                actual_metrics = subset["Metric"].tolist()
                expected_metrics = [
                    m + " relative to annotation1" for m in requested_metrics
                ]
                assert all(metric in actual_metrics for metric in expected_metrics)

        # AvgValue for the annotation being used as normalization factor should be 1
        assert (df.loc[df["Annotation"] == "annotation1", "AvgValue"] == 1).all()
        # Validate that the AvgValue for the annotation being used for normalization is greater than 1
        assert (df.loc[df["Annotation"] == "annotation2", "AvgValue"] > 1).all()


# ============================================================================
# Output prefix tests
# ============================================================================


@nsight.analyze.kernel(
    configs=[(32,)],
    runs=1,
    output="quiet",
    output_prefix="/tmp/test_output_prefix/test_prefix_",
)
def output_prefix_func(n: int) -> None:
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")
    with nsight.annotate("test"):
        _ = a + b


def test_parameter_output_prefix() -> None:
    """Test that output_prefix creates directories and files with correct prefix."""
    output_dir = "/tmp/test_output_prefix"
    try:
        result = output_prefix_func()

        if result is not None:
            expected_files = [
                "/tmp/test_output_prefix/test_prefix_ncu-output-output_prefix_func-0.ncu-rep",
                "/tmp/test_output_prefix/test_prefix_ncu-output-output_prefix_func-0.log",
            ]

            for file_path in expected_files:
                assert os.path.exists(
                    file_path
                ), f"Expected file not found: {file_path}"
    finally:
        # Cleanup after test
        if "NSPY_NCU_PROFILE" not in os.environ:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)


# ----------------------------------------------------------------------------


@pytest.mark.parametrize("output_csv", [True, False])  # type: ignore[untyped-decorator]
def test_parameter_output_csv(output_csv: bool) -> None:
    """Test the output_csv parameter to control CSV file generation."""
    output_dir = "/tmp/test_output_csv/"

    try:

        @nsight.analyze.kernel(
            configs=[(42, 23)],
            output_prefix=f"{output_dir}test_",
            output_csv=output_csv,
        )
        def output_csv_func(x: int, y: int) -> None:
            _simple_kernel_impl(x, y, annotation=f"output_csv={output_csv}")

        # Run the profiling
        profile_output = output_csv_func()

        # Verify that ProfileResults is returned (even if CSV is not dumped)
        if "NSPY_NCU_PROFILE" not in os.environ:

            # Check for CSV files based on output_csv value
            csv_files = [
                f"{output_dir}test_profiled_data-output_csv_func-0.csv",
                f"{output_dir}test_processed_data-output_csv_func-0.csv",
            ]

            for file_path in csv_files:
                if output_csv:
                    assert os.path.exists(
                        file_path
                    ), f"CSV file should exist when output_csv=True: {file_path}"
                else:
                    assert not os.path.exists(
                        file_path
                    ), f"CSV file should not exist when output_csv=False: {file_path}"

            # NCU report files should always exist regardless of output_csv
            ncu_files = [
                f"{output_dir}test_ncu-output-output_csv_func-0.ncu-rep",
                f"{output_dir}test_ncu-output-output_csv_func-0.log",
            ]

            for file_path in ncu_files:
                assert os.path.exists(
                    file_path
                ), f"NCU file should always exist: {file_path}"
    finally:
        # Cleanup after test
        if "NSPY_NCU_PROFILE" not in os.environ:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)


# ============================================================================
# Ignore kernel list tests
# ============================================================================


@pytest.mark.parametrize("ignore_kernel_list", [None, ["vector_multiply"]])  # type: ignore[untyped-decorator]
def test_parameter_ignore_kernel_list(ignore_kernel_list: None | list[str]) -> None:
    """Test the ignore_kernel_list parameter to filter out specific kernels"""

    @nsight.analyze.kernel(
        configs=[(1024,)], runs=1, output="quiet", ignore_kernel_list=ignore_kernel_list
    )
    def ignore_kernel_func(n: int) -> None:

        device = Device()
        device.set_current()
        s = device.create_stream()

        program = Program(CUDA_KERNEL_CODE, "c++")
        module = program.compile("cubin")
        kernel_add = module.get_kernel("vector_add")
        kernel_mul = module.get_kernel("vector_multiply")

        # Allocate device memory
        size_bytes = n * 4  # float32
        d_a = device.allocate(size_bytes)
        d_b = device.allocate(size_bytes)
        d_c = device.allocate(size_bytes)

        # Launch both kernels in one annotation
        with nsight.annotate(f"test_kernels_{ignore_kernel_list}"):
            block = 256
            grid = (size_bytes + block - 1) // block
            config = LaunchConfig(grid=grid, block=block)

            # This kernel should be ignored
            launch(s, config, kernel_add, d_a, d_b, d_c, n)

            # This kernel should be profiled
            launch(s, config, kernel_mul, d_a, d_b, d_c, n)

        # Synchronize
        device.sync()

    if ignore_kernel_list is None:
        with pytest.raises(
            RuntimeError,
            match=f"More than one.*kernel.*launched within the test_kernels_{ignore_kernel_list} annotation",
        ):
            profile_output = ignore_kernel_func()
    else:

        profile_output = ignore_kernel_func()
        kernel_names = profile_output.to_dataframe()["Kernel"].to_list()
        ignored_kernel_name = ignore_kernel_list[0]
        assert ignored_kernel_name not in kernel_names


# ============================================================================
# Clock control tests
# ============================================================================


@pytest.mark.parametrize("clock_control", ["base", "none", "invalid_value"])  # type: ignore[untyped-decorator]
def test_parameter_clock_control(
    clock_control: Literal["base", "none", "invalid_value"],
) -> None:
    """Test the clock_control parameter to control GPU clock locking."""

    if clock_control == "invalid_value":
        with pytest.raises(ValueError):
            nsight.analyze.kernel(clock_control=clock_control)(lambda x: x)  # type: ignore[call-overload]
    else:

        @nsight.analyze.kernel(
            configs=[(100, 100)],
            runs=1,
            output="quiet",
            clock_control=clock_control,
        )
        def clock_control_func(x: int, y: int) -> None:
            _simple_kernel_impl(x, y, "test_clock_control")

        # Should complete successfully regardless of clock_control value
        profile_output = clock_control_func()
        assert profile_output is not None, "ProfileResults should be returned"


# ============================================================================
# Cache control tests
# ============================================================================


@pytest.mark.parametrize("cache_control", ["all", "none", "invalid_value"])  # type: ignore[untyped-decorator]
def test_parameter_cache_control(
    cache_control: Literal["all", "none", "invalid_value"],
) -> None:
    """Test the cache_control parameter to control cache flush behavior."""

    if cache_control == "invalid_value":
        with pytest.raises(ValueError):
            nsight.analyze.kernel(cache_control=cache_control)(lambda x: x)  # type: ignore[call-overload]
    else:

        @nsight.analyze.kernel(
            configs=[(100, 100)],
            runs=1,
            output="quiet",
            cache_control=cache_control,
        )
        def cache_control_func(x: int, y: int) -> None:
            _simple_kernel_impl(x, y, "test_cache_control")

        # Should complete successfully regardless of cache_control value
        profile_output = cache_control_func()
        assert profile_output is not None, "ProfileResults should be returned"


# ============================================================================
# Thermal control tests
# ============================================================================


@pytest.mark.parametrize("thermal_control", [True, False])  # type: ignore[untyped-decorator]
def test_parameter_thermal_control(thermal_control: bool) -> None:
    """Test the thermal_control parameter to control thermal waiting."""

    @nsight.analyze.kernel(
        configs=[(100, 100)],
        runs=1,
        output="quiet",
        thermal_control=thermal_control,
    )
    def thermal_control_func(x: int, y: int) -> None:
        _simple_kernel_impl(x, y, "test_thermal_control")

    # Should complete successfully regardless of thermal_control value
    profile_output = thermal_control_func()
    assert profile_output is not None, "ProfileResults should be returned"


# ============================================================================
# replay_mode parameter tests
# ============================================================================


@pytest.mark.parametrize(
    "replay_mode",
    [
        "kernel",  # kernel mode should fail with multiple kernels
        "range",  # range mode should handle multiple kernels
        "invalid_value",  # range mode should handle invalid values
    ],
)  # type: ignore[untyped-decorator]
def test_parameter_replay_mode(
    replay_mode: Literal["kernel", "range", "invalid_value"],
) -> None:
    """Test that replay_mode parameter correctly handles multiple kernels."""

    if replay_mode == "invalid_value":
        with pytest.raises(ValueError):
            nsight.analyze.kernel(replay_mode=replay_mode)(lambda x: x)  # type: ignore[call-overload]
    else:

        @nsight.analyze.kernel(
            configs=[(1024,)],
            runs=1,
            output="quiet",
            replay_mode=replay_mode,
        )
        def multiple_kernels_replay_test(n: int) -> None:
            device = Device()
            device.set_current()
            s = device.create_stream()

            program = Program(CUDA_KERNEL_CODE, "c++")
            module = program.compile("cubin")
            kernel_add = module.get_kernel("vector_add")
            kernel_mul = module.get_kernel("vector_multiply")

            # Allocate device memory
            size_bytes = n * 4  # float32
            d_a = device.allocate(size_bytes)
            d_b = device.allocate(size_bytes)
            d_c = device.allocate(size_bytes)

            # Launch multiple kernels in the same annotation
            with nsight.annotate("test_replay"):
                block = 256
                grid = (size_bytes + block - 1) // block
                config = LaunchConfig(grid=grid, block=block)

                # Launch two different kernels
                launch(s, config, kernel_add, d_a, d_b, d_c, n)
                launch(s, config, kernel_mul, d_a, d_b, d_c, n)

            # Synchronize
            device.sync()

        if replay_mode == "kernel":
            # replay_mode="kernel" should raise RuntimeError for multiple kernels
            with pytest.raises(
                RuntimeError,
                match="More than one.*kernel.*launched within the test_replay annotation",
            ):
                multiple_kernels_replay_test()
        else:
            # replay_mode="range" should handle multiple kernels successfully
            profile_output = multiple_kernels_replay_test()
            df = profile_output.to_dataframe()
            assert df["Kernel"][0] == "range"


# ============================================================================
# derive_metric parameter tests
# ============================================================================


def _compute_custom_metric(time_ns: float, x: int, y: int) -> float:
    """Transform time in nanoseconds to a custom metric based on matrix size."""
    # Custom formula: operations per second (arbitrary for testing)
    operations = x * y
    time_s = time_ns / 1e9
    return operations / time_s if time_s > 0 else 0.0


@pytest.mark.parametrize(
    "derive_metric_func,expected_name",
    [
        (_compute_custom_metric, "_compute_custom_metric"),
        (lambda time_ns, x, y: (x * y) / (time_ns / 1e9) / 1e9, "<lambda>"),
    ],
)  # type: ignore[untyped-decorator]
def test_parameter_derive_metric(derive_metric_func: Any, expected_name: str) -> None:
    """Test the derive_metric parameter to transform collected metrics."""

    @nsight.analyze.kernel(
        configs=[(100, 100), (200, 200)],
        runs=2,
        output="quiet",
        derive_metric=derive_metric_func,
    )
    def profiled_func(x: int, y: int) -> None:
        _simple_kernel_impl(x, y, "test_derive_metric")

    # Run profiling
    profile_output = profiled_func()
    assert profile_output is not None, "ProfileResults should be returned"

    # Verify the transformed metric is present
    df = profile_output.to_dataframe()
    assert "Transformed" in df.columns, "Transformed column should exist"
    assert (
        df["Transformed"].iloc[0] == expected_name
    ), f"Transformed column should show '{expected_name}'"

    # Verify the metric values are transformed (should be positive numbers)
    assert "AvgValue" in df.columns, "AvgValue column should exist"
    assert all(df["AvgValue"] > 0), "All derived metric values should be positive"

    # Verify we have results for both configs
    assert len(df) == 2, "Should have results for 2 configurations"


# ============================================================================
# output parameter test
# ============================================================================


@pytest.mark.parametrize("output", ["quiet", "progress", "verbose", "invalid_value"])  # type: ignore[untyped-decorator]
def test_parameter_output(
    capsys: pytest.CaptureFixture,
    output: Literal["quiet", "progress", "verbose", "invalid_value"],
) -> None:
    if output == "invalid_value":
        with pytest.raises(
            ValueError, match="output must be 'quiet', 'progress' or 'verbose'"
        ):
            nsight.analyze.kernel(output=output)(lambda x: x)  # type: ignore[call-overload]

    else:

        @nsight.analyze.kernel(configs=[(100, 100), (200, 200)], runs=2, output=output)
        def profiled_func(x: int, y: int) -> None:
            _simple_kernel_impl(x, y, "test_parameter_output")

        # Run profiling
        profile_output = profiled_func()
        assert profile_output is not None, "ProfileResults should be returned"

        # Check output
        captured = capsys.readouterr()
        if output == "quiet":
            assert len(captured.out) == 0, "stdout should be empty for output='quiet'"

        # TODO:"progress" and "verbose" modes


# ============================================================================
# metrics parameter test
# ============================================================================


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "metrics, expected_result",
    [
        pytest.param(
            [
                "invalid_value",
            ],
            "invalid_single",
            id="invalid_single",
        ),
        pytest.param(
            [
                "sm__warps_launched.sum",
            ],
            "valid_single",
            id="valid_single",
        ),
        pytest.param(
            [
                "smsp__inst_executed.sum",
                "smsp__inst_issued.sum",
            ],
            "invalid_multiple",
            id="invalid_multiple",
        ),
    ],
)
def test_parameter_metric(metrics: Sequence[str], expected_result: str) -> None:

    @nsight.analyze.plot(filename="plot.png", ylabel="Instructions")
    @nsight.analyze.kernel(configs=[(100, 100), (200, 200)], runs=2, metrics=metrics)
    def profiled_func(x: int, y: int) -> None:
        _simple_kernel_impl(x, y, "test_parameter_metric")

    # Run profiling
    if expected_result == "invalid_single":
        with pytest.raises(
            exceptions.ProfilerException,
            match=(
                rf"Invalid value \['{metrics[0]}'\] for 'metrics' parameter for nsight.analyze.kernel()"
            ),
        ):
            profiled_func()
    elif expected_result == "valid_single":
        profile_output = profiled_func()
        df = profile_output.to_dataframe()

        # Checking if the dataframe has the right metric name
        assert (
            df["Metric"] == metrics[0]
        ).all(), f"Invalid metric name {df.loc[df['Metric'] != metrics[0], 'Metric'].iloc[0]} found in output dataframe"

        # Checking if the metric values are valid
        assert (
            df["AvgValue"].notna() & df["AvgValue"] > 0
        ).all(), f"Invalid AvgValue for metric {metrics}"
    elif expected_result == "invalid_multiple":
        with pytest.raises(
            ValueError,
            match=(
                f"Cannot visualize {len(metrics)} > 1 metrics with the @nsight.analyze.plot decorator."
            ),
        ):
            profiled_func()
