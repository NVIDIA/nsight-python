# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nsight.transformation.aggregate_data (no GPU required)."""

from typing import Any

import pandas as pd

from nsight import transformation


def _one_arg(config: Any) -> None:
    """A function with a single config parameter (num_args == 1)."""
    return None


def _raw_df(config_values: list[Any], *, param_name: str = "config") -> pd.DataFrame:
    """Synthetic raw frame; the config column is last, as aggregate_data expects."""
    n = len(config_values)
    data: dict[str, Any] = {
        "Annotation": ["matmul"] * n,
        "Metric": ["gpu__time_duration.sum"] * n,
        "Value": [float(i + 1) for i in range(n)],
        "Kernel": ["ampere_sgemm_128x128"] * n,
        param_name: config_values,
    }
    return pd.DataFrame(data)


def test_dict_config_single_row_does_not_crash() -> None:
    # Regression: a single dict-config row used to crash groupby().agg() with
    # "unhashable type: 'dict'" because the dict was never stringified.
    dims = {"M": 512, "K": 512, "N": 512}
    result = transformation.aggregate_data(_raw_df([dims]), _one_arg, None, False)
    assert len(result) == 1
    assert result["config"].iloc[0] == str(dims)
    assert result["NumRuns"].iloc[0] == 1
    assert result["AvgValue"].iloc[0] == 1.0


def test_dict_config_independent_of_run_count() -> None:
    # Bumping runs only masked the bug (it added rows); every count must work.
    dims = {"M": 256, "K": 256, "N": 256}
    for runs in (1, 2, 5):
        result = transformation.aggregate_data(
            _raw_df([dims] * runs), _one_arg, None, False
        )
        assert len(result) == 1
        assert result["NumRuns"].iloc[0] == runs


def test_distinct_dict_configs_stay_distinct() -> None:
    # Stringification is per-row, so different dicts remain different groups.
    d1, d2 = {"M": 512}, {"M": 1024}
    result = transformation.aggregate_data(_raw_df([d1, d2]), _one_arg, None, False)
    assert len(result) == 2
    assert set(result["config"]) == {str(d1), str(d2)}


def test_list_config_single_row_does_not_crash() -> None:
    # Lists are unhashable too.
    shape = [512, 512]
    result = transformation.aggregate_data(_raw_df([shape]), _one_arg, None, False)
    assert len(result) == 1
    assert result["config"].iloc[0] == str(shape)


def test_numeric_config_keeps_native_dtype() -> None:
    # Sortable, hashable configs must not be needlessly coerced to strings.
    result = transformation.aggregate_data(_raw_df([128]), _one_arg, None, False)
    assert result["config"].iloc[0] == 128
    assert pd.api.types.is_integer_dtype(result["config"])
