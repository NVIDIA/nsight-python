# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental Nsight Python APIs.

APIs in this module may change without the compatibility guarantees of the
stable package surface.
"""

from collections.abc import Callable
from typing import Any, TypeVar, overload

from nsight.info_collector import CollectionScope

_Collector = TypeVar("_Collector", bound=Callable[..., Any])


def _canonical_scope(scope: CollectionScope | str) -> str:
    """Return a validated string representation of a collection scope."""
    if isinstance(scope, CollectionScope):
        return scope.value
    try:
        return CollectionScope(scope).value
    except (TypeError, ValueError) as exc:
        valid_scopes = ", ".join(repr(item.value) for item in CollectionScope)
        raise ValueError(
            f"Invalid collection scope {scope!r}. Expected one of {valid_scopes} "
            "or a CollectionScope."
        ) from exc


@overload
def collect(
    _func: _Collector,
    *,
    scope: CollectionScope | str,
) -> _Collector: ...


@overload
def collect(
    _func: None = None,
    *,
    scope: CollectionScope | str,
) -> Callable[[_Collector], _Collector]: ...


def collect(
    _func: _Collector | None = None,
    *,
    scope: CollectionScope | str,
) -> _Collector | Callable[[_Collector], _Collector]:
    """Mark a function as a custom information collector.

    Args:
        _func: Function to decorate. This is normally supplied by decorator
            syntax, but direct calls are also supported.
        scope: When the function is called. Supported values are
            :class:`nsight.CollectionScope` members or their string values:
            ``"once"``, ``"config"``, ``"run"``, and ``"annotation"``.

    Examples:
        Decorator syntax::

            @nsight.experimental.collect(scope=nsight.CollectionScope.RUN)
            def gpu_temperature(*config_args):
                return read_gpu_temperature()

        Direct-call syntax::

            gpu_temperature = nsight.experimental.collect(
                gpu_temperature, scope="run"
            )
    """
    canonical_scope = _canonical_scope(scope)

    def decorator(func: _Collector) -> _Collector:
        if not callable(func):
            raise TypeError("collect can only decorate a callable")
        setattr(func, "_nsight_collector_scope", canonical_scope)
        setattr(func, "_nsight_collector_name", func.__name__)
        return func

    if _func is None:
        return decorator
    return decorator(_func)
