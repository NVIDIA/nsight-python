# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private execution identities shared by the collectors and annotations."""

import functools
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import ParamSpec

P = ParamSpec("P")


@dataclass(frozen=True)
class Invocation:
    annotation: str
    config_index: int
    run_index: int


@dataclass
class Session:
    token: str
    invocations: dict[str, Invocation]
    current_run: tuple[int, int] | None = None
    seen: set[str] = field(default_factory=set)


_lock = threading.RLock()
_active: Session | None = None


def capture(func: Callable[P, None]) -> Callable[P, dict[str, Invocation]]:
    """Keep identities until collection ends; always clear live session state."""

    @functools.wraps(func)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> dict[str, Invocation]:
        global _active
        with _lock:
            if _active is not None:
                raise RuntimeError("Overlapping profiling sessions are not supported")
            session = Session(uuid.uuid4().hex, {})
            _active = session
        try:
            func(*args, **kwargs)
            return session.invocations
        finally:
            with _lock:
                _active = None

    return wrapped


@contextmanager
def run(config_index: int, run_index: int) -> Iterator[None]:
    with _lock:
        assert _active is not None
        _active.current_run = (config_index, run_index)
        _active.seen.clear()
    try:
        yield
    finally:
        with _lock:
            assert _active is not None
            _active.current_run = None


def marker_name(annotation: str) -> str:
    """Keep the readable name, but distinguish every profiled execution.

    Process-wide state also covers annotations entered on joined worker threads.
    The collector maps the exact internal name back to the public annotation;
    user strings are never parsed or interpreted as internal identifiers.
    """
    with _lock:
        if _active is None or _active.current_run is None:
            return annotation
        config_index, run_index = _active.current_run
        invocation = Invocation(annotation, config_index, run_index)
        if annotation in _active.seen:
            raise ValueError(f"Annotation {annotation!r} was already used in this run")
        _active.seen.add(annotation)
        marker = f"{annotation} [nspy:{_active.token}:{len(_active.invocations)}]"
        _active.invocations[marker] = invocation
        return marker
