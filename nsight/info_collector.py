# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Info collector utilities for collecting custom information during profiling.

This module provides the core data structures for custom information collectors
that can be passed to the @nsight.analyze.kernel decorator.
"""

import inspect
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nsight import exceptions


class CollectionScope(Enum):
    """
    Defines when a collector should be invoked during profiling.

    Attributes:
        ONCE: Collector is invoked once at the start of profiling.
              Use for static information like driver version, CUDA version, etc.
        CONFIG: Collector is invoked once for each configuration.
                Use for configuration-derived information that does not change
                between repeated runs.
        RUN: Collector is invoked once for every run of each configuration.
             Use for dynamic information like clocks, temperature, memory usage, etc.
        ANNOTATION: Collector is invoked at the start of each annotated region.
                    Use for per-annotation state like cache status, active threads, etc.
    """

    ONCE = "once"
    CONFIG = "config"
    RUN = "run"
    ANNOTATION = "annotation"


@dataclass
class InfoCollector:
    """
    Data class representing an information collector.

    Attributes:
        name: The column name in the resulting DataFrame.
        callback: Function to call to collect the information.
        scope: When this collector should be invoked (ONCE, CONFIG, RUN, or
            ANNOTATION).

    Example:
        >>> from nsight.info_collector import InfoCollector, CollectionScope
        >>>
        >>> # Create a ONCE scope collector
        >>> def get_version():
        ...     return "1.0.0"
        >>> version_collector = InfoCollector("Version", get_version, CollectionScope.ONCE)
        >>>
        >>> # Create a CONFIG scope collector
        >>> def sum_args(*args):
        ...     return sum(args)
        >>> sum_collector = InfoCollector("Sum", sum_args, CollectionScope.CONFIG)
        >>>
        >>> # Create a RUN scope collector
        >>> def current_temperature(*args):
        ...     return 42
        >>> temperature_collector = InfoCollector(
        ...     "Temperature", current_temperature, CollectionScope.RUN
        ... )
        >>>
        >>> # Create an ANNOTATION scope collector
        >>> def get_annotation_info(annotation_name, *config_args):
        ...     return f"{annotation_name}_{config_args[0]}"
        >>> ann_collector = InfoCollector("AnnInfo", get_annotation_info, CollectionScope.ANNOTATION)
    """

    name: str
    callback: Callable[..., Any]
    scope: CollectionScope

    def collect(self, *args: Any) -> Any:
        """
        Collect information by invoking the callback.

        Args:
            *args: Arguments passed to the callback. ONCE-scope collectors
                receive no arguments, CONFIG- and RUN-scope collectors receive
                the configuration arguments, and ANNOTATION-scope collectors
                receive the annotation name followed by the configuration
                arguments.

        Returns:
            The collected information.
        """
        if self.scope == CollectionScope.ONCE:
            # For ONCE scope, we don't pass any arguments
            return self.callback()
        return self.callback(*args)


# Column names that the profiler itself produces in the result DataFrame.
# A custom collector may not reuse any of these, or it would silently overwrite
# real profiling data (built-in columns) or aggregation outputs.
RESERVED_COLUMN_NAMES = frozenset(
    {
        # Built-in columns from extraction.extract_df_from_report
        "Annotation",
        "Value",
        "Metric",
        "Kernel",
        "GPU",
        "Host",
        "ComputeClock",
        "MemoryClock",
        "Unit",
        # Aggregation outputs from transformation.aggregate_data
        "AvgValue",
        "StdDev",
        "MinValue",
        "MaxValue",
        "NumRuns",
        "CI95_Lower",
        "CI95_Upper",
        "RelativeStdDevPct",
        "StableMeasurement",
        "Geomean",
        "Normalized",
        "NormalizationValue",
    }
)


def validate_collectors(
    collectors: Sequence[Tuple[str, Callable[..., Any], str]],
    func: Optional[Callable[..., Any]] = None,
) -> None:
    """Validate normalized ``(name, callback, scope)`` collector tuples.

    Raises a clear ``ProfilerException`` (fail-fast, before any profiling work)
    when a collector name would silently clobber another column:

    * duplicate collector names (one would overwrite the other),
    * a name colliding with a reserved profiler column (see
      ``RESERVED_COLUMN_NAMES``),
    * a name colliding with one of the decorated function's parameter names
      (the parameter column would overwrite the collector column).

    Args:
        collectors: Normalized collector tuples.
        func: The decorated function, used to detect parameter-name collisions.
            Optional so the check can run before ``func`` is known.
    """
    seen: set[str] = set()
    for entry in collectors:
        name = entry[0]
        if name in seen:
            raise exceptions.ProfilerException(
                f"Duplicate info collector name '{name}'. Each collector must "
                "have a unique name (the function name becomes the column name)."
            )
        seen.add(name)
        if name in RESERVED_COLUMN_NAMES:
            raise exceptions.ProfilerException(
                f"Info collector name '{name}' collides with a reserved profiler "
                f"column. Reserved names: {sorted(RESERVED_COLUMN_NAMES)}."
            )

    if func is not None:
        param_names = {
            p.name
            for p in inspect.signature(func).parameters.values()
            if p.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        }
        for name in seen:
            if name in param_names:
                raise exceptions.ProfilerException(
                    f"Info collector name '{name}' collides with a parameter of "
                    f"'{func.__name__}'. Rename the collector so it does not shadow "
                    "a function argument column."
                )


def bind_args_to_signature(
    callback: Callable[..., Any],
    leading: Sequence[Any],
    config: Sequence[Any],
) -> List[Any]:
    """Compute the positional args to pass to a collector callback.

    ``leading`` are mandatory leading args (e.g. the annotation name for
    annotation-scope collectors); ``config`` is the (possibly default-padded)
    config tuple. A collector that declares ``*args`` receives the full config;
    a fixed-arity collector receives only as many leading config values as its
    remaining positional parameters, so default-padded or extra config values do
    not cause a spurious ``TypeError`` (which was previously swallowed, silently
    nulling the column).
    """
    try:
        params = list(inspect.signature(callback).parameters.values())
    except (TypeError, ValueError):
        # Builtins / C callables with no introspectable signature: pass all.
        return list(leading) + list(config)

    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
        return list(leading) + list(config)

    positional_params = [
        p
        for p in params
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    remaining = max(0, len(positional_params) - len(leading))
    return list(leading) + list(config[:remaining])


# Collectors that have already triggered a failure warning this session, so we
# warn once per (scope, name) instead of once per run (which would be spammy).
_warned_failures: set[str] = set()


def reset_collector_failure_warnings() -> None:
    """Reset warning de-duplication at the start of a profiling session."""
    _warned_failures.clear()


def warn_collector_failure(name: str, scope: str, exc: BaseException) -> None:
    """Surface a collector failure regardless of the profiler's output mode.

    Collector exceptions are swallowed (the column degrades to None), but a
    collector that fails on every run would otherwise be completely invisible
    in the default/quiet output modes. This emits a single RuntimeWarning the
    first time a given collector fails, so the failure is never silent.
    """
    key = f"{scope}:{name}"
    if key in _warned_failures:
        return
    _warned_failures.add(key)
    warnings.warn(
        f"Info collector '{name}' ({scope} scope) raised "
        f"{type(exc).__name__}: {exc}. Its column will be None for failed runs.",
        category=RuntimeWarning,
        stacklevel=2,
    )


# Process-global state for annotation collectors and collected data.
# Profiling runs exactly one session at a time, and annotated regions may be
# entered from worker threads. This state is therefore intentionally
# process-global rather than thread-local:
# a thread-local store silently lost all annotation data whenever the kernel
# was launched from a thread other than the one running run_profile_session.
@dataclass
class _AnnotationCollectorState:
    """Process-global state for annotation collectors."""

    collectors: List[InfoCollector] = field(default_factory=list)
    current_config: tuple[Any, ...] = field(default_factory=tuple)
    collected_data: List[Dict[str, Any]] = field(default_factory=list)
    output_detailed: bool = False


_state = _AnnotationCollectorState()


def _get_state() -> _AnnotationCollectorState:
    """Get the process-global annotation-collector state."""
    return _state


def set_annotation_collectors(
    collectors: List[InfoCollector],
    current_config: tuple[Any, ...],
    output_detailed: bool = False,
) -> None:
    """
    Set the annotation collectors and current config for this process.

    This is called by run_profile_session before running each configuration.

    Args:
        collectors: List of ANNOTATION scope collectors.
        current_config: The current configuration tuple.
        output_detailed: Whether to print detailed output.
    """
    state = _get_state()
    state.collectors = collectors
    state.current_config = current_config
    state.output_detailed = output_detailed


def collect_for_annotation(annotation_name: str) -> Dict[str, Any]:
    """
    Collect information for an annotation.

    This is called by the annotate context manager when entering an annotation.

    Args:
        annotation_name: Name of the annotation being entered.

    Returns:
        Dictionary mapping collector names to collected values.
    """
    state = _get_state()
    if not state.collectors:
        return {}

    collected = {}

    for collector in state.collectors:
        try:
            # Pass annotation name and current config to the collector, trimming
            # extra (default-padded) config values for fixed-arity collectors.
            collector_args = bind_args_to_signature(
                collector.callback, (annotation_name,), state.current_config
            )
            value = collector.collect(*collector_args)
            collected[collector.name] = value
            if state.output_detailed:
                print(
                    f"[NSIGHT-PYTHON] Collected {collector.name} ({annotation_name}): {value}"
                )
        except Exception as e:
            collected[collector.name] = None
            warn_collector_failure(collector.name, "annotation", e)
            if state.output_detailed:
                print(
                    f"[NSIGHT-PYTHON] Warning: Failed to collect '{collector.name}' for annotation '{annotation_name}': {e}"
                )

    # Store for later retrieval
    state.collected_data.append(
        {
            "annotation": annotation_name,
            "config": state.current_config,
            "data": collected,
        }
    )

    return collected


def get_collected_annotation_data() -> List[Dict[str, Any]]:
    """
    Get all collected annotation data for the current profiling run.

    Returns:
        List of collected annotation data dictionaries (a copy).
    """
    state = _get_state()
    # Return a copy to avoid reference issues when clearing for next run
    return state.collected_data.copy()


def clear_annotation_data() -> None:
    """Clear all collected annotation data for the current profiling run."""
    state = _get_state()
    state.collected_data.clear()
