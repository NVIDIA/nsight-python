# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for custom info collectors feature.

These tests verify the complete data flow from collection to DataFrame output,
including edge cases and bugs that have been discovered.
"""

from typing import Any

from cuda.core import (  # pylint: disable=import-error
    Device,
    LaunchConfig,
    Program,
    ProgramOptions,
    launch,
)

import nsight  # pylint: disable=import-error

# Simple CUDA kernels for testing - no memory allocation to avoid extra internal kernels
_kernel_code = r"""
__global__ void dummy_kernel_add() {
    // Simple no-op kernel for testing
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    (void)idx;  // Suppress unused variable warning
}

__global__ void dummy_kernel_mul() {
    // Another simple no-op kernel for testing
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    (void)idx;  // Suppress unused variable warning
}
"""

_compiled_module: Any = None


def _get_compiled_module() -> Any:
    """Lazily compile and cache the CUDA module."""
    global _compiled_module  # pylint: disable=global-statement
    if _compiled_module is None:
        program_options = ProgramOptions(std="c++17")
        prog = Program(_kernel_code, code_type="c++", options=program_options)
        _compiled_module = prog.compile(
            "cubin", name_expressions=("dummy_kernel_add", "dummy_kernel_mul")
        )
    return _compiled_module


def _launch_kernel_add(n: int) -> None:
    """Launch a simple add kernel (no memory allocation)."""
    del n  # Not needed for dummy kernel
    module = _get_compiled_module()
    device = Device()
    device.set_current()
    stream = device.create_stream()

    # Launch kernel with minimal grid/block
    config = LaunchConfig(grid=1, block=256)
    kernel = module.get_kernel("dummy_kernel_add")
    launch(stream, config, kernel)

    stream.sync()


def _launch_kernel_mul(n: int) -> None:
    """Launch a simple mul kernel (no memory allocation)."""
    del n  # Not needed for dummy kernel
    module = _get_compiled_module()
    device = Device()
    device.set_current()
    stream = device.create_stream()

    # Launch kernel with minimal grid/block
    config = LaunchConfig(grid=1, block=256)
    kernel = module.get_kernel("dummy_kernel_mul")
    launch(stream, config, kernel)

    stream.sync()


# ============================================================================
# Test 1: annotation_scope_reference_bug
# ============================================================================

_annotation_counter_ref_bug = {"count": 0}


@nsight.experimental.collect(scope=nsight.CollectionScope.ANNOTATION)
def _exec_counter_ref_bug(
    annotation_name: str, *config_args: Any
) -> int:  # pylint: disable=unused-argument
    """Collector for reference bug test."""
    _annotation_counter_ref_bug["count"] += 1
    return _annotation_counter_ref_bug["count"]


@nsight.analyze.kernel(
    configs=[(512,), (1024,)],
    runs=3,
    verbosity=nsight.VerbosityLevel.SILENT,
    info_collectors=[_exec_counter_ref_bug],
)
def benchmark_annotation_scope_reference_bug(n: int) -> None:
    """Benchmark for testing annotation-scope reference bug."""
    with nsight.annotate("add"):
        _launch_kernel_add(n)

    with nsight.annotate("mul"):
        _launch_kernel_mul(n)


def test_annotation_scope_reference_bug() -> None:
    """
    Test that annotation-scope collectors don't have reference bugs.

    This is a regression test for a bug where get_collected_annotation_data()
    was returning a reference to the thread-local list, causing all runs to
    share the same list that was being cleared between runs.
    """
    results = benchmark_annotation_scope_reference_bug()

    if results is None:
        return  # No profiling data was returned.

    df = results.to_dataframe()

    # Check that we have the exec_counter columns
    assert "_exec_counter_ref_bug_Min" in df.columns
    assert "_exec_counter_ref_bug_Max" in df.columns
    assert "_exec_counter_ref_bug_Avg" in df.columns

    # For the first config (512), second annotation (mul):
    # Runs 1, 2, 3: mul gets exec_counter values 2, 4, 6
    # Min should be 2
    first_mul = df[(df["n"] == 512) & (df["Annotation"] == "mul")]
    assert len(first_mul) == 1
    assert first_mul["_exec_counter_ref_bug_Min"].iloc[0] == 2
    assert first_mul["_exec_counter_ref_bug_Max"].iloc[0] == 6

    # For the first config (512), first annotation (add):
    # Runs 1, 2, 3: add gets exec_counter values 1, 3, 5
    # Min should be 1
    first_add = df[(df["n"] == 512) & (df["Annotation"] == "add")]
    assert len(first_add) == 1
    assert first_add["_exec_counter_ref_bug_Min"].iloc[0] == 1
    assert first_add["_exec_counter_ref_bug_Max"].iloc[0] == 5

    # For the second config (1024), annotations should continue counting
    # add: 7, 9, 11 (min=7, max=11)
    # mul: 8, 10, 12 (min=8, max=12)
    second_add = df[(df["n"] == 1024) & (df["Annotation"] == "add")]
    assert second_add["_exec_counter_ref_bug_Min"].iloc[0] == 7
    assert second_add["_exec_counter_ref_bug_Max"].iloc[0] == 11

    second_mul = df[(df["n"] == 1024) & (df["Annotation"] == "mul")]
    assert second_mul["_exec_counter_ref_bug_Min"].iloc[0] == 8
    assert second_mul["_exec_counter_ref_bug_Max"].iloc[0] == 12


# ============================================================================
# Test 2: once_scope_collectors
# ============================================================================

_once_call_count = {"count": 0}


@nsight.experimental.collect(scope=nsight.CollectionScope.ONCE)
def _static_version() -> str:
    """Once-scope collector for testing."""
    _once_call_count["count"] += 1
    return "v1.0.0"


@nsight.analyze.kernel(
    configs=[(512,), (1024,)],
    runs=3,
    verbosity=nsight.VerbosityLevel.SILENT,
    info_collectors=[_static_version],
)
def benchmark_once_scope_collectors(n: int) -> None:
    """Benchmark for testing once-scope collectors."""
    with nsight.annotate("add"):
        _launch_kernel_add(n)


def test_once_scope_collectors() -> None:
    """Test that once-scope collectors are called exactly once and constant across all rows."""
    results = benchmark_once_scope_collectors()

    if results is None:
        return  # No profiling data was returned.

    df = results.to_dataframe()

    # Value should be constant across all rows.
    assert "_static_version" in df.columns
    assert all(df["_static_version"] == "v1.0.0")
    assert df["_static_version"].nunique() == 1


# ============================================================================
# Test 3: config_scope_collectors
# ============================================================================

_collected_configs: list[int] = []


@nsight.experimental.collect(scope=nsight.CollectionScope.CONFIG)
def _track_config(n: int) -> int:
    """Config-scope collector for testing."""
    _collected_configs.append(n)
    return n * 2


@nsight.analyze.kernel(
    configs=[(512,), (1024,)],
    runs=2,
    verbosity=nsight.VerbosityLevel.SILENT,
    info_collectors=[_track_config],
)
def benchmark_config_scope_collectors(n: int) -> None:
    """Benchmark for testing config-scope collectors."""
    with nsight.annotate("add"):
        _launch_kernel_add(n)


def test_config_scope_collectors() -> None:
    """Test that config collectors are called once per config with config args."""
    _collected_configs.clear()
    results = benchmark_config_scope_collectors()

    if results is None:
        return  # No profiling data was returned.

    df = results.to_dataframe()

    # Results should be aggregated.
    assert "_track_config_Avg" in df.columns
    assert "_track_config_Min" in df.columns
    assert "_track_config_Max" in df.columns

    # For n=512, value should be 1024 (512*2)
    row_512 = df[df["n"] == 512]
    assert row_512["_track_config_Avg"].iloc[0] == 1024

    # For n=1024, value should be 2048 (1024*2)
    row_1024 = df[df["n"] == 1024]
    assert row_1024["_track_config_Avg"].iloc[0] == 2048
    assert _collected_configs == [512, 1024]


# ============================================================================
# Test 3b: custom collectors combined with derive_metric (regression for H1)
# ============================================================================


@nsight.experimental.collect(scope=nsight.CollectionScope.ONCE)
def _derive_static_version() -> str:
    """Once-scope collector used alongside derive_metric."""
    return "v9.9.9"


@nsight.experimental.collect(scope=nsight.CollectionScope.CONFIG)
def _derive_config_value(n: int) -> int:
    """Config-scope numeric collector used alongside derive_metric."""
    return n * 3


def _derive_double(time_ns: float, n: int) -> tuple[float, str]:
    """Derived metric: twice the kernel time (value, unit)."""
    return (time_ns * 2.0, "ns2")


@nsight.analyze.kernel(
    configs=[(512,), (1024,)],
    runs=2,
    verbosity=nsight.VerbosityLevel.SILENT,
    derive_metric=_derive_double,
    info_collectors=[_derive_static_version, _derive_config_value],
)
def benchmark_collectors_with_derive_metric(n: int) -> None:
    """Benchmark combining custom info collectors with a derive_metric."""
    with nsight.annotate("add"):
        _launch_kernel_add(n)


def test_collectors_with_derive_metric() -> None:
    """Regression (H1): custom-info columns must be populated on derived-metric
    rows, not NaN/None. Previously transformed_df_data omitted custom_info_arrays
    so every derived-metric row got NaN for collectors."""
    results = benchmark_collectors_with_derive_metric()

    if results is None:
        return  # No profiling data was returned.

    df = results.to_dataframe()

    # The derived metric forms its own Metric rows (named after the function).
    derived_rows = df[df["Metric"] == "_derive_double"]
    assert not derived_rows.empty, "expected derived-metric rows"

    # Once-scope string column: present and constant (NOT None/NaN) on derived rows.
    assert "_derive_static_version" in df.columns
    assert derived_rows["_derive_static_version"].notna().all()
    assert (derived_rows["_derive_static_version"] == "v9.9.9").all()

    # Config-scope numeric column: aggregated and non-NaN on derived rows.
    assert "_derive_config_value_Avg" in df.columns
    assert derived_rows["_derive_config_value_Avg"].notna().all()
    # n=512 -> collector returns 512*3 = 1536.
    derived_512 = derived_rows[derived_rows["n"] == 512]
    assert derived_512["_derive_config_value_Avg"].iloc[0] == 1536


# ============================================================================
# Test 4: annotation_scope_collectors
# ============================================================================

_collected_annotation_data: list[tuple[str, int]] = []


@nsight.experimental.collect(scope=nsight.CollectionScope.ANNOTATION)
def _track_annotation(annotation_name: str, n: int) -> str:
    """Annotation-scope collector for testing."""
    _collected_annotation_data.append((annotation_name, n))
    return f"{annotation_name}_{n}"


@nsight.analyze.kernel(
    configs=[(512,), (1024,)],
    runs=2,
    verbosity=nsight.VerbosityLevel.SILENT,
    info_collectors=[_track_annotation],
)
def benchmark_annotation_scope_collectors(n: int) -> None:
    """Benchmark for testing annotation-scope collectors."""
    with nsight.annotate("add"):
        _launch_kernel_add(n)

    with nsight.annotate("mul"):
        _launch_kernel_mul(n)


def test_annotation_scope_collectors() -> None:
    """Test that annotation-scope collectors are called per annotation and receive both annotation name and config."""
    results = benchmark_annotation_scope_collectors()

    if results is None:
        return  # No profiling data was returned.

    df = results.to_dataframe()

    # Non-numeric annotation data should appear as-is (first value).
    assert "_track_annotation" in df.columns

    # Check specific values
    add_512 = df[(df["n"] == 512) & (df["Annotation"] == "add")]
    assert add_512["_track_annotation"].iloc[0] == "add_512"

    mul_1024 = df[(df["n"] == 1024) & (df["Annotation"] == "mul")]
    assert mul_1024["_track_annotation"].iloc[0] == "mul_1024"


# ============================================================================
# Test 5: mixed_scopes
# ============================================================================


@nsight.experimental.collect(scope=nsight.CollectionScope.ONCE)
def _once_value() -> str:
    """Once collector for mixed scopes test."""
    return "once"


@nsight.experimental.collect(scope=nsight.CollectionScope.CONFIG)
def _config_value(n: int) -> int:
    """Config collector for mixed scopes test."""
    return n


@nsight.experimental.collect(scope=nsight.CollectionScope.ANNOTATION)
def _annotation_value(
    annotation_name: str, n: int
) -> int:  # pylint: disable=unused-argument
    """Annotation collector for mixed scopes test."""
    return len(annotation_name)


@nsight.analyze.kernel(
    configs=[(512,), (1024,)],
    runs=2,
    verbosity=nsight.VerbosityLevel.SILENT,
    info_collectors=[_once_value, _config_value, _annotation_value],
)
def benchmark_mixed_scopes(n: int) -> None:
    """Benchmark for testing all three scopes together."""
    with nsight.annotate("add"):
        _launch_kernel_add(n)


def test_mixed_scopes() -> None:
    """Test using collectors from all three scopes together."""
    results = benchmark_mixed_scopes()

    if results is None:
        return  # No profiling data was returned.

    df = results.to_dataframe()

    # All three collectors should produce columns
    assert "_once_value" in df.columns
    assert "_config_value_Avg" in df.columns
    assert "_annotation_value_Avg" in df.columns

    # Once value should be constant
    assert all(df["_once_value"] == "once")

    # Config value should match n
    assert df[df["n"] == 512]["_config_value_Avg"].iloc[0] == 512
    assert df[df["n"] == 1024]["_config_value_Avg"].iloc[0] == 1024

    # Annotation value should be len("add") = 3
    assert all(df["_annotation_value_Avg"] == 3)


# ============================================================================
# Test 6: numeric_vs_non_numeric_aggregation
# ============================================================================


@nsight.experimental.collect(scope=nsight.CollectionScope.RUN)
def _numeric_collector(n: int) -> int:  # pylint: disable=unused-argument
    """Numeric collector for aggregation test."""
    import random

    return random.randint(100, 200)


@nsight.experimental.collect(scope=nsight.CollectionScope.RUN)
def _string_collector(n: int) -> str:
    """String collector for aggregation test."""
    return f"config_{n}"


@nsight.analyze.kernel(
    configs=[(512,)],
    runs=5,
    verbosity=nsight.VerbosityLevel.SILENT,
    info_collectors=[_numeric_collector, _string_collector],
)
def benchmark_numeric_vs_non_numeric_aggregation(n: int) -> None:
    """Benchmark for testing numeric vs non-numeric aggregation."""
    with nsight.annotate("add"):
        _launch_kernel_add(n)


def test_numeric_vs_non_numeric_aggregation() -> None:
    """Test that numeric collectors are aggregated while non-numeric ones take first value."""
    results = benchmark_numeric_vs_non_numeric_aggregation()

    if results is None:
        return  # No profiling data was returned.

    df = results.to_dataframe()

    # Numeric collector should have aggregation columns
    assert "_numeric_collector_Avg" in df.columns
    assert "_numeric_collector_Std" in df.columns
    assert "_numeric_collector_Min" in df.columns
    assert "_numeric_collector_Max" in df.columns

    # Non-numeric collector should appear as-is
    assert "_string_collector" in df.columns
    assert "_string_collector_Avg" not in df.columns

    # Check that numeric values are within expected range
    row = df.iloc[0]
    assert 100 <= row["_numeric_collector_Min"] <= 200
    assert 100 <= row["_numeric_collector_Max"] <= 200
    assert 100 <= row["_numeric_collector_Avg"] <= 200

    # String value should be the first collected value
    assert row["_string_collector"] == "config_512"


# ============================================================================
# Test 7: collector_error_handling
# ============================================================================

_error_success_count = {"count": 0}


@nsight.experimental.collect(scope=nsight.CollectionScope.RUN)
def _failing_collector(n: int) -> int:
    """Collector that fails for some configs."""
    if n == 512:
        raise ValueError("Simulated collection failure")
    return n * 2


@nsight.experimental.collect(scope=nsight.CollectionScope.RUN)
def _success_collector(n: int) -> int:
    """Collector that always succeeds."""
    _error_success_count["count"] += 1
    return n * 2


@nsight.analyze.kernel(
    configs=[(512,), (1024,)],
    runs=1,
    verbosity=nsight.VerbosityLevel.SILENT,
    info_collectors=[_failing_collector, _success_collector],
)
def benchmark_collector_error_handling(n: int) -> None:
    """Benchmark for testing collector error handling."""
    with nsight.annotate("add"):
        _launch_kernel_add(n)


def test_collector_error_handling() -> None:
    """Test that collector failures don't crash profiling."""
    # Should not raise despite failing_collector error
    results = benchmark_collector_error_handling()

    if results is None:
        return  # No profiling data was returned.

    df = results.to_dataframe()

    # Success collector returns n*2, aggregated per config.
    assert "_success_collector_Avg" in df.columns
    assert df[df["n"] == 512]["_success_collector_Avg"].iloc[0] == 1024
    assert df[df["n"] == 1024]["_success_collector_Avg"].iloc[0] == 2048

    # Failing collector raised for n=512 -> NaN there, but succeeded for
    # n=1024 -> 2048. Profiling itself must not crash.
    assert "_failing_collector_Avg" in df.columns
    assert df[df["n"] == 512]["_failing_collector_Avg"].isna().iloc[0]
    assert df[df["n"] == 1024]["_failing_collector_Avg"].iloc[0] == 2048


# ============================================================================
# Test 8: multiple_annotations_per_run
# ============================================================================

_annotation_order: list[str] = []


@nsight.experimental.collect(scope=nsight.CollectionScope.ANNOTATION)
def _track_order(
    annotation_name: str, n: int
) -> int:  # pylint: disable=unused-argument
    """Annotation-scope collector for tracking order."""
    _annotation_order.append(annotation_name)
    return len(_annotation_order)


@nsight.analyze.kernel(
    configs=[(512,)],
    runs=2,
    verbosity=nsight.VerbosityLevel.SILENT,
    info_collectors=[_track_order],
)
def benchmark_multiple_annotations_per_run(n: int) -> None:
    """Benchmark for testing multiple annotations."""
    with nsight.annotate("first"):
        _launch_kernel_add(n)

    with nsight.annotate("second"):
        _launch_kernel_mul(n)

    with nsight.annotate("third"):
        _launch_kernel_add(n)


def test_multiple_annotations_per_run() -> None:
    """Test that annotation-scope collectors work correctly with multiple annotations."""
    results = benchmark_multiple_annotations_per_run()

    if results is None:
        return  # No profiling data was returned.

    df = results.to_dataframe()

    # Should have 3 rows (one per annotation).
    assert len(df) == 3

    # Each annotation should have different min/max values showing execution order
    first = df[df["Annotation"] == "first"]
    second = df[df["Annotation"] == "second"]
    third = df[df["Annotation"] == "third"]

    # first runs at positions 1 and 4
    assert first["_track_order_Min"].iloc[0] == 1
    assert first["_track_order_Max"].iloc[0] == 4

    # second runs at positions 2 and 5
    assert second["_track_order_Min"].iloc[0] == 2
    assert second["_track_order_Max"].iloc[0] == 5

    # third runs at positions 3 and 6
    assert third["_track_order_Min"].iloc[0] == 3
    assert third["_track_order_Max"].iloc[0] == 6


# ============================================================================
# Test 9: decorator_based_api
# ============================================================================


@nsight.experimental.collect(scope=nsight.CollectionScope.ONCE)
def _my_once_collector() -> str:
    """Once collector for decorator API test."""
    return "once_value"


@nsight.experimental.collect(scope=nsight.CollectionScope.CONFIG)
def _my_config_collector(n: int) -> int:
    """Config collector for decorator API test."""
    return n


@nsight.experimental.collect(scope=nsight.CollectionScope.RUN)
def _my_run_collector(n: int) -> int:
    """Run collector for decorator API test."""
    return n


@nsight.experimental.collect(scope=nsight.CollectionScope.ANNOTATION)
def _my_annotation_collector(
    annotation_name: str, n: int
) -> str:  # pylint: disable=unused-argument
    """Annotation collector for decorator API test."""
    return annotation_name


@nsight.analyze.kernel(
    configs=[(512,)],
    runs=1,
    verbosity=nsight.VerbosityLevel.SILENT,
    info_collectors=[
        _my_once_collector,
        _my_config_collector,
        _my_run_collector,
        _my_annotation_collector,
    ],
)
def benchmark_decorator_based_api(n: int) -> None:
    """Benchmark for testing the decorator-based API."""
    with nsight.annotate("add"):
        _launch_kernel_add(n)


def test_decorator_based_api() -> None:
    """Test the decorator-based API for defining collectors."""
    results = benchmark_decorator_based_api()

    if results is None:
        return  # No profiling data was returned.

    df = results.to_dataframe()

    # Column names should match function names
    assert "_my_once_collector" in df.columns
    assert "_my_config_collector_Avg" in df.columns
    assert "_my_run_collector_Avg" in df.columns
    assert "_my_annotation_collector" in df.columns


# ============================================================================
# Test 10: empty_collectors_list
# ============================================================================


@nsight.analyze.kernel(
    configs=[(512,)],
    runs=1,
    verbosity=nsight.VerbosityLevel.SILENT,
    info_collectors=[],
)
def benchmark_empty_collectors(n: int) -> None:
    """Benchmark with empty collectors list."""
    with nsight.annotate("add"):
        _launch_kernel_add(n)


@nsight.analyze.kernel(configs=[(512,)], runs=1, verbosity=nsight.VerbosityLevel.SILENT)
def benchmark_none_collectors(n: int) -> None:
    """Benchmark with no collectors specified."""
    with nsight.annotate("add"):
        _launch_kernel_add(n)


def test_empty_collectors_list() -> None:
    """Test that profiling works with empty or no collectors."""
    # Both should work without errors
    results1 = benchmark_empty_collectors()
    results2 = benchmark_none_collectors()

    # Some collector implementations may return no profiling data.
    if results1 is not None:
        assert results1 is not None
    if results2 is not None:
        assert results2 is not None


# ============================================================================
# Test 11: column_ordering
# ============================================================================


@nsight.experimental.collect(scope=nsight.CollectionScope.ONCE)
def _z_collector() -> str:
    """Once collector with name starting with 'z' for ordering test."""
    return "z_value"


@nsight.experimental.collect(scope=nsight.CollectionScope.CONFIG)
def _a_collector(n: int) -> int:
    """Config collector with name starting with 'a' for ordering test."""
    return n * 2


@nsight.analyze.kernel(
    configs=[(512,)],
    runs=1,
    verbosity=nsight.VerbosityLevel.SILENT,
    info_collectors=[_z_collector, _a_collector],
)
def benchmark_column_ordering(n: int) -> None:
    """Benchmark for testing column ordering."""
    with nsight.annotate("add"):
        _launch_kernel_add(n)


def test_column_ordering() -> None:
    """Test that custom info columns are present in the DataFrame."""
    results = benchmark_column_ordering()

    if results is None:
        return  # No profiling data was returned.

    df = results.to_dataframe()

    # Verify all expected columns exist
    assert "_z_collector" in df.columns
    assert "_a_collector_Avg" in df.columns
    assert "n" in df.columns

    # Verify the values are correct
    assert df["_z_collector"].iloc[0] == "z_value"
    assert df["_a_collector_Avg"].iloc[0] == 1024  # 512 * 2
    assert df["n"].iloc[0] == 512
