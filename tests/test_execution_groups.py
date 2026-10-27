# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
import pytest

from nsight import _session, extraction, transformation, utils
from nsight.collection import cupti


def benchmark(size: int) -> None:
    pass


def action(value: float | None, name: str = "kernel") -> utils.ActionData:
    return utils.ActionData(
        name, None if value is None else np.array([value]), 1, 1, "GPU", ["ns"]
    )


def extract(
    data: dict[str, list[utils.ActionData]],
    identities: dict[str, _session.Invocation],
    **kwargs: Any,
) -> Any:
    return extraction.extract_df_from_data(
        data,
        ["time"],
        [(10,), (20,)],
        2,
        benchmark,
        None,
        utils.VerbosityLevel.SILENT,
        invocations=identities,
        **kwargs,
    )


@pytest.mark.parametrize("counts", [(2, 3), (2, 4)])  # type: ignore[untyped-decorator]
def test_variable_counts(counts: tuple[int, int]) -> None:
    identities = {
        f"c{c}r{r}": _session.Invocation("public", c, r)
        for c in range(2)
        for r in range(2)
    }
    data = {
        key: [action(float(inv.config_index + 1))] * counts[inv.config_index]
        for key, inv in identities.items()
    }
    # Collector delivery order must not assign samples to a different config.
    df = extract(
        dict(reversed(list(data.items()))),
        identities,
        combine_kernel_metrics=lambda a, b: a + b,
    )
    assert df.Annotation.tolist() == ["public"] * 4
    assert df.Value.tolist() == [counts[0]] * 2 + [2 * counts[1]] * 2
    result = transformation.aggregate_data(
        df, benchmark, None, utils.VerbosityLevel.SILENT
    )
    assert result.NumRuns.tolist() == [2, 2]


def test_variable_counts_within_config_and_reducer_order() -> None:
    identities = {
        "a": _session.Invocation("public", 0, 0),
        "b": _session.Invocation("public", 0, 1),
    }
    df = extract(
        {"a": [action(1), action(2)], "b": [action(3), action(4), action(5)]},
        identities,
        combine_kernel_metrics=lambda a, b: 10 * a + b,
    )
    assert df.Value.tolist() == [12, 345]


def test_missing_annotation_keeps_run_index_aligned() -> None:
    # Extraction uses this absolute index to select custom info for a skipped run.
    groups = extraction._group_samples(
        {"second": [action(9)]},
        [(10,), (20,)],
        2,
        {"second": _session.Invocation("public", 1, 1)},
    )
    index, config, kernels = groups["public"][0]
    assert index == 3
    assert config == (20,)
    assert kernels[0].values is not None
    assert kernels[0].values.tolist() == [9]


def test_empty_or_fully_filtered_invocation_is_explicit() -> None:
    with pytest.raises(RuntimeError, match="annotation 'public', config 1, run 0"):
        extract({}, {"empty": _session.Invocation("public", 1, 0)})


def test_unknown_execution_rejected() -> None:
    with pytest.raises(RuntimeError, match="unknown annotation executions"):
        extract({"unexpected": [action(1)]}, {})


def test_multiple_kernels_still_require_reducer() -> None:
    with pytest.raises(RuntimeError, match="Supply combine_kernel_metrics"):
        extract(
            {"a": [action(1), action(2)]}, {"a": _session.Invocation("public", 0, 0)}
        )


def test_partial_failure_remains_missing() -> None:
    df = extract(
        {"a": [action(1), action(None, utils.DUMMY_KERNEL_NAME)]},
        {"a": _session.Invocation("public", 0, 0)},
        combine_kernel_metrics=lambda a, b: a + b,
    )
    assert df.Value.isna().all()


def test_session_identities_and_cleanup() -> None:
    names: list[str] = []

    def collect() -> dict[str, _session.Invocation]:
        with _session.capture() as identities:
            for c in range(2):
                with _session.run(c, 0):
                    names.append(_session.marker_name("same [nspy:looks-internal]"))
        return identities

    first = collect()
    second = collect()
    assert not first.keys() & second.keys()
    assert [x.config_index for x in first.values()] == [0, 1]
    assert all(x.annotation == "same [nspy:looks-internal]" for x in first.values())
    assert _session.marker_name("outside") == "outside"

    def failing() -> None:
        with _session.capture():
            with _session.run(0, 0):
                _session.marker_name("a")
                raise ValueError("failed")

    with pytest.raises(ValueError, match="failed"):
        failing()
    assert _session.marker_name("outside") == "outside"
    collect()


def test_cupti_keeps_execution_labels() -> None:
    # The same public annotation has distinct labels for each invocation.
    markers = [
        cupti._MarkerRecord(1, 0, True, "first"),
        cupti._MarkerRecord(1, 10, False, ""),
        cupti._MarkerRecord(2, 20, True, "second"),
        cupti._MarkerRecord(2, 30, False, ""),
    ]
    launches = [cupti._LaunchRecord(i, t) for i, t in enumerate([1, 2, 21, 22, 23])]
    kernels = [cupti._KernelRecord(i, "k", 1) for i in range(5)]
    attributed = cupti._attribute_kernels(markers, launches, kernels)
    data: dict[str, list[utils.ActionData]] = {}
    for label, name, duration in attributed:
        data.setdefault(label, []).append(action(float(duration), name))
    df = extract(
        data,
        {
            "first": _session.Invocation("public", 0, 0),
            "second": _session.Invocation("public", 1, 0),
        },
        combine_kernel_metrics=lambda a, b: a + b,
    )
    assert df.Value.tolist() == [2, 3]


def test_scalar_generator_configs_keep_values_aligned() -> None:
    # Collectors materialize generators without normalizing each scalar.
    configs: Any = list(x for x in (10, 20))
    df = extraction.extract_df_from_data(
        {"a": [action(1)], "b": [action(2)]},
        ["time"],
        configs,
        1,
        benchmark,
        None,
        utils.VerbosityLevel.SILENT,
        invocations={
            "a": _session.Invocation("public", 0, 0),
            "b": _session.Invocation("public", 1, 0),
        },
    )
    assert df["size"].tolist() == [10, 20]
    assert df.Value.tolist() == [1, 2]


def test_annotation_outside_run_fails_at_entry_and_session_recovers() -> None:
    from nsight import annotate

    for after_run in (False, True):
        with _session.capture() as identities:
            if after_run:
                with _session.run(0, 0):
                    _session.marker_name("inside")
            with pytest.raises(RuntimeError, match="outside a profiled config/run"):
                with annotate("outside"):
                    pytest.fail("annotation body must not execute")
            assert all(x.annotation != "outside" for x in identities.values())
    assert _session.marker_name("outside") == "outside"


def test_worker_annotation_after_run_has_clear_error() -> None:
    import threading

    errors: list[str] = []

    def worker() -> None:
        try:
            _session.marker_name("late worker")
        except RuntimeError as exc:
            errors.append(str(exc))

    with _session.capture():
        with _session.run(0, 0):
            pass
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
    assert len(errors) == 1
    assert "late worker" in errors[0]
    assert "join worker threads" in errors[0]


def test_session_context_keeps_profile_call_return_value() -> None:
    # The execution mapping must not replace metadata returned by collection.
    returned_info = [{"config_info": {"value": object()}}]

    def profile_call() -> list[dict[str, Any]]:
        with _session.run(0, 0):
            _session.marker_name("kernel")
        return returned_info

    with _session.capture() as identities:
        result = profile_call()
    assert result is returned_info
    assert list(identities.values()) == [_session.Invocation("kernel", 0, 0)]


def test_collector_context_covers_post_run_teardown() -> None:
    @_session.capture()
    def collector() -> object:
        value = object()
        with _session.run(0, 0):
            _session.marker_name("inside")
        assert list(_session.get_invocations().values()) == [
            _session.Invocation("inside", 0, 0)
        ]
        with pytest.raises(RuntimeError, match="outside a profiled config/run"):
            _session.marker_name("during synchronization")
        return value

    assert collector() is not None
    assert _session.marker_name("after collection") == "after collection"
