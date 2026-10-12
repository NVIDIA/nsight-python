# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle
from pathlib import Path
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


def test_missing_annotation_keeps_custom_info_aligned(tmp_path: Path) -> None:
    prefix = str(tmp_path / "info-")
    with open(prefix + "collected_info.pkl", "wb") as f:
        pickle.dump([{"config_info": {"run_value": i}} for i in range(4)], f)
    df = extract(
        {"second": [action(9)]},
        {"second": _session.Invocation("public", 1, 1)},
        info_collectors_list=[("run_value", lambda: 0, "run")],
        info_prefix=prefix,
    )
    assert df["size"].tolist() == [20]
    assert df.run_value.tolist() == [3]


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

    @_session.capture
    def collect() -> None:
        for c in range(2):
            with _session.run(c, 0):
                names.append(_session.marker_name("same [nspy:looks-internal]"))

    first = collect()
    second = collect()
    assert not first.keys() & second.keys()
    assert [x.config_index for x in first.values()] == [0, 1]
    assert all(x.annotation == "same [nspy:looks-internal]" for x in first.values())
    assert _session.marker_name("outside") == "outside"

    @_session.capture
    def failing() -> None:
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
