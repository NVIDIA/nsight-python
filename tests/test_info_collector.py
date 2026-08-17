# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the info_collector module.
"""

import builtins
import pickle
import threading
import warnings
from typing import Any

import pytest

import nsight
from nsight import exceptions, info_collector
from nsight.collection import core
from nsight.info_collector import CollectionScope, InfoCollector


class TestInfoCollector:
    """Tests for InfoCollector dataclass."""

    def test_collect_once_scope(self) -> None:
        """Test collecting with ONCE scope."""

        def callback() -> str:
            return "test_value"

        collector = InfoCollector(
            name="TestCollector", callback=callback, scope=CollectionScope.ONCE
        )

        result = collector.collect()
        assert result == "test_value"

    def test_collect_config_scope(self) -> None:
        """Test collecting with CONFIG scope."""

        def callback(x: int, y: int) -> int:
            return x + y

        collector = InfoCollector(
            name="TestCollector", callback=callback, scope=CollectionScope.CONFIG
        )

        result = collector.collect(10, 20)
        assert result == 30

    def test_collect_config_scope_with_args(self) -> None:
        """Test CONFIG scope collector receives config args."""

        def callback(*args: int) -> int:
            return sum(args)

        collector = InfoCollector(
            name="SumCollector", callback=callback, scope=CollectionScope.CONFIG
        )

        result = collector.collect(1, 2, 3, 4, 5)
        assert result == 15

    def test_collect_run_scope_with_args(self) -> None:
        """Test RUN scope collector receives config args."""

        collector = InfoCollector(
            name="RunCollector",
            callback=lambda value: value * 2,
            scope=CollectionScope.RUN,
        )

        assert collector.collect(21) == 42


class TestExperimentalCollect:
    """Tests for the unified experimental collector decorator."""

    def test_decorator_accepts_enum_scope(self) -> None:
        @nsight.experimental.collect(scope=CollectionScope.CONFIG)
        def collector(value: int) -> int:
            return value

        assert collector._nsight_collector_scope == "config"  # type: ignore[attr-defined]
        assert collector._nsight_collector_name == "collector"  # type: ignore[attr-defined]

    def test_direct_call_accepts_string_scope(self) -> None:
        def callback() -> int:
            return 1

        collector = nsight.experimental.collect(callback, scope="once")

        assert collector is callback
        assert collector._nsight_collector_scope == "once"  # type: ignore[attr-defined]

    def test_rejects_invalid_scope(self) -> None:
        with pytest.raises(ValueError, match="Invalid collection scope"):
            nsight.experimental.collect(scope="invalid")


def test_profile_session_scope_invocation_counts(tmp_path: Any) -> None:
    """CONFIG runs once per config, while RUN and ANNOTATION run repeatedly."""
    calls: dict[str, list[Any]] = {
        "once": [],
        "config": [],
        "run": [],
        "annotation": [],
    }

    def collect_once() -> str:
        calls["once"].append(None)
        return "static"

    def collect_config(value: int) -> int:
        calls["config"].append(value)
        return value

    def collect_run(value: int) -> int:
        calls["run"].append(value)
        return value

    def collect_annotation(annotation_name: str, value: int) -> str:
        calls["annotation"].append((annotation_name, value))
        return annotation_name

    def kernel(value: int) -> None:
        info_collector.collect_for_annotation("region")

    core.run_profile_session(
        kernel,
        [(1,), (2,)],
        runs=3,
        verbosity=nsight.VerbosityLevel.SILENT,
        thermal_mode="off",
        info_collectors_list=[
            ("once", collect_once, "once"),
            ("config", collect_config, "config"),
            ("run", collect_run, "run"),
            ("annotation", collect_annotation, "annotation"),
        ],
        output_prefix=f"{tmp_path}/",
    )

    assert calls["once"] == [None]
    assert calls["config"] == [1, 2]
    assert calls["run"] == [1, 1, 1, 2, 2, 2]
    assert calls["annotation"] == [
        ("region", 1),
        ("region", 1),
        ("region", 1),
        ("region", 2),
        ("region", 2),
        ("region", 2),
    ]
    assert info_collector.collect_for_annotation("after_profile") == {}


def test_profile_session_resets_failure_warnings(tmp_path: Any) -> None:
    """The same failing collector warns once in each profiling session."""

    def failing_collector() -> None:
        raise RuntimeError("expected failure")

    def kernel() -> None:
        pass

    with pytest.warns(RuntimeWarning, match="expected failure") as warnings_list:
        for _ in range(2):
            core.run_profile_session(
                kernel,
                [()],
                runs=1,
                verbosity=nsight.VerbosityLevel.SILENT,
                thermal_mode="off",
                info_collectors_list=[("failing", failing_collector, "run")],
                output_prefix=f"{tmp_path}/",
            )

    assert len(warnings_list) == 2


def test_non_picklable_collector_value_is_sanitized(tmp_path: Any) -> None:
    """Only the non-picklable value is replaced before writing the file."""
    info_file = tmp_path / "collected_info.pkl"
    collected_info: list[dict[str, Any]] = [
        {
            "config": (),
            "run_idx": 0,
            "once_info": {"good": "value", "bad": lambda: None},
            "config_info": {},
            "annotation_info": [],
        }
    ]

    with pytest.warns(RuntimeWarning, match="not picklable"):
        saved = core._save_collected_info(
            str(info_file), collected_info, nsight.VerbosityLevel.SILENT
        )

    assert saved
    with info_file.open("rb") as file:
        saved_info = pickle.load(file)
    assert saved_info[0]["once_info"] == {"good": "value", "bad": None}


def test_collected_info_io_failure_is_reported_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disk write failure is not reported as a non-picklable value."""

    def fail_open(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise OSError("disk full")

    monkeypatch.setattr(builtins, "open", fail_open)

    with pytest.warns(
        RuntimeWarning, match="Failed to save.*disk full"
    ) as warnings_list:
        saved = core._save_collected_info(
            "/unwritable/collected_info.pkl",
            [],
            nsight.VerbosityLevel.SILENT,
        )

    assert not saved
    assert "not picklable" not in str(warnings_list[0].message)


class TestCollectorIntegration:
    """Integration tests for info collectors."""

    def test_real_world_scenario(self) -> None:
        """Test a realistic scenario with multiple collectors."""

        # Simulate collecting static info
        def get_driver_version() -> str:
            return "555.42"

        def get_cuda_version() -> str:
            return "12.6"

        # Simulate collecting dynamic info
        def get_temperature(*config_args: int) -> int:
            # In real scenario, this would query nvidia-smi
            # For testing, we'll just use the first arg
            return 50 + config_args[0] if config_args else 50

        def compute_size(*config_args: int) -> int:
            if len(config_args) >= 2:
                return config_args[0] * config_args[1]
            return 0

        # Create collector objects directly
        once_collectors = [
            InfoCollector("DriverVersion", get_driver_version, CollectionScope.ONCE),
            InfoCollector("CUDAVersion", get_cuda_version, CollectionScope.ONCE),
        ]
        config_collectors = [
            InfoCollector("Temperature", get_temperature, CollectionScope.CONFIG),
            InfoCollector("ComputedSize", compute_size, CollectionScope.CONFIG),
        ]

        # Collect once info
        once_data = {c.name: c.collect() for c in once_collectors}

        assert once_data["DriverVersion"] == "555.42"
        assert once_data["CUDAVersion"] == "12.6"

        # Collect config info for multiple configs
        configs = [(512, 512), (1024, 1024), (2048, 2048)]

        config_data = []
        for config in configs:
            row_data = {c.name: c.collect(*config) for c in config_collectors}
            config_data.append(row_data)

        # Verify config-specific data
        assert config_data[0]["Temperature"] == 50 + 512
        assert config_data[1]["Temperature"] == 50 + 1024
        assert config_data[2]["Temperature"] == 50 + 2048

        assert config_data[0]["ComputedSize"] == 512 * 512
        assert config_data[1]["ComputedSize"] == 1024 * 1024
        assert config_data[2]["ComputedSize"] == 2048 * 2048


class TestValidateCollectors:
    """Validation of collector names (fail-fast before profiling)."""

    def test_rejects_duplicate_names(self) -> None:
        with pytest.raises(exceptions.ProfilerException, match="Duplicate"):
            info_collector.validate_collectors(
                [
                    ("dup", lambda: 1, "once"),
                    ("dup", lambda: 2, "once"),
                ]
            )

    def test_rejects_cross_scope_duplicate(self) -> None:
        with pytest.raises(exceptions.ProfilerException, match="Duplicate"):
            info_collector.validate_collectors(
                [
                    ("name", lambda: 1, "once"),
                    ("name", lambda *a: 2, "annotation"),
                ]
            )

    def test_rejects_reserved_column_name(self) -> None:
        for reserved in ("Kernel", "GPU", "Value", "Annotation", "AvgValue"):
            with pytest.raises(exceptions.ProfilerException, match="reserved"):
                info_collector.validate_collectors([(reserved, lambda: 1, "once")])

    def test_rejects_aggregation_output_name(self) -> None:
        # Columns added by transformation.aggregate_data must also be reserved,
        # otherwise a collector named after one is silently clobbered.
        for reserved in ("Geomean", "CI95_Lower", "Normalized", "StableMeasurement"):
            with pytest.raises(exceptions.ProfilerException, match="reserved"):
                info_collector.validate_collectors([(reserved, lambda *a: 1, "config")])

    def test_rejects_collision_with_function_param(self) -> None:
        def kernel(n, dtype):  # type: ignore[no-untyped-def]
            return None

        with pytest.raises(exceptions.ProfilerException, match="parameter"):
            info_collector.validate_collectors(
                [("n", lambda *a: 1, "config")], func=kernel
            )

    def test_accepts_valid_collectors(self) -> None:
        def kernel(n, dtype):  # type: ignore[no-untyped-def]
            return None

        # No exception expected.
        info_collector.validate_collectors(
            [
                ("driver", lambda: "1.0", "once"),
                ("temp", lambda *a: 1, "config"),
            ],
            func=kernel,
        )


class TestBindArgsToSignature:
    """Config-to-collector argument binding (handles default-padded configs)."""

    def test_var_positional_receives_full_config(self) -> None:
        assert info_collector.bind_args_to_signature(lambda *a: a, (), (1, 2, 3)) == [
            1,
            2,
            3,
        ]

    def test_fixed_arity_trims_extra_padded_values(self) -> None:
        # Kernel had a defaulted 3rd param; config padded to len 3, but the
        # collector only declares one argument -> it must receive just that one.
        assert info_collector.bind_args_to_signature(lambda n: n, (), (1, 2, 3)) == [1]

    def test_annotation_leading_arg_with_fixed_collector(self) -> None:
        assert info_collector.bind_args_to_signature(
            lambda annotation_name, n: None, ("add",), (1, 2, 3)
        ) == ["add", 1]

    def test_annotation_leading_arg_with_var_positional(self) -> None:
        assert info_collector.bind_args_to_signature(
            lambda annotation_name, *cfg: None, ("add",), (1, 2)
        ) == ["add", 1, 2]


class TestAnnotationStateMachine:
    """GPU-free coverage of the thread/process-global annotation state machine."""

    def test_get_collected_returns_independent_copy_across_runs(self) -> None:
        """Regression: get_collected_annotation_data() must return a copy so a
        later run's clear()/collect() cannot retroactively empty an earlier
        run's snapshot (the bug the .copy() in get_collected_annotation_data
        guards against)."""
        collector = InfoCollector(
            "counter", lambda name, *cfg: name, CollectionScope.ANNOTATION
        )

        # Run 1
        info_collector.set_annotation_collectors([collector], (512,))
        info_collector.clear_annotation_data()
        info_collector.collect_for_annotation("add")
        run1 = info_collector.get_collected_annotation_data()

        # Run 2
        info_collector.clear_annotation_data()
        info_collector.collect_for_annotation("mul")
        run2 = info_collector.get_collected_annotation_data()

        assert len(run1) == 1 and run1[0]["annotation"] == "add"
        assert len(run2) == 1 and run2[0]["annotation"] == "mul"

    def test_off_thread_collection_sees_collectors(self) -> None:
        """Process-global (not thread-local) state: a region entered from a
        worker thread still sees collectors configured on the main thread.
        With the old threading.local store this returned an empty dict."""
        collector = InfoCollector(
            "t", lambda name, *cfg: name, CollectionScope.ANNOTATION
        )
        info_collector.set_annotation_collectors([collector], (256,))
        info_collector.clear_annotation_data()

        captured: dict[str, object] = {}

        def worker() -> None:
            captured["collected"] = info_collector.collect_for_annotation("on_worker")

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        assert captured["collected"] == {"t": "on_worker"}
        data = info_collector.get_collected_annotation_data()
        assert any(d["annotation"] == "on_worker" for d in data)

    def test_annotation_failure_is_isolated_and_warns(self) -> None:
        info_collector.reset_collector_failure_warnings()

        def boom(annotation_name, *config_args):  # type: ignore[no-untyped-def]
            raise RuntimeError("collector blew up")

        collector = InfoCollector("boom", boom, CollectionScope.ANNOTATION)
        info_collector.set_annotation_collectors([collector], ())
        info_collector.clear_annotation_data()

        with pytest.warns(RuntimeWarning, match="boom"):
            result = info_collector.collect_for_annotation("region")

        assert result["boom"] is None


class TestCollectorFailureWarning:
    """warn_collector_failure surfaces failures regardless of output verbosity."""

    def test_warns_once_per_collector(self) -> None:
        info_collector.reset_collector_failure_warnings()
        with pytest.warns(RuntimeWarning, match="flaky"):
            info_collector.warn_collector_failure("flaky", "config", ValueError("x"))
        # Second failure of the same collector should not warn again.
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning would raise
            info_collector.warn_collector_failure("flaky", "config", ValueError("y"))

    def test_reset_warns_again_in_next_session(self) -> None:
        info_collector.reset_collector_failure_warnings()
        with pytest.warns(RuntimeWarning, match="flaky"):
            info_collector.warn_collector_failure("flaky", "run", ValueError("x"))

        info_collector.reset_collector_failure_warnings()
        with pytest.warns(RuntimeWarning, match="flaky"):
            info_collector.warn_collector_failure("flaky", "run", ValueError("y"))
