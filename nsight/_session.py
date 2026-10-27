# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private execution identities shared by the collectors and annotations."""

import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


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


@contextmanager
def capture() -> Iterator[dict[str, Invocation]]:
    """Track execution identities without changing the profiled call's return value."""
    global _active
    with _lock:
        if _active is not None:
            raise RuntimeError("Overlapping profiling sessions are not supported")
        session = Session(uuid.uuid4().hex, {})
        _active = session
    try:
        yield session.invocations
    finally:
        with _lock:
            _active = None


@contextmanager
def run(config_index: int, run_index: int) -> Iterator[None]:
    with _lock:
        session = _active
        if session is not None:
            session.current_run = (config_index, run_index)
            session.seen.clear()
    try:
        yield
    finally:
        with _lock:
            if session is not None:
                session.current_run = None


def marker_name(annotation: str) -> str:
    """Keep the readable name, but distinguish every profiled execution.

    Process-wide state also covers annotations entered on joined worker threads.
    The collector maps the exact internal name back to the public annotation;
    user strings are never parsed or interpreted as internal identifiers.
    """
    with _lock:
        if _active is None:
            return annotation
        if _active.current_run is None:
            raise RuntimeError(
                f"Annotation {annotation!r} was entered outside a profiled config/run. "
                "Enter annotations inside the profiled function, and join worker "
                "threads before that function returns. Info-collector callbacks "
                "cannot enter annotations."
            )
        config_index, run_index = _active.current_run
        invocation = Invocation(annotation, config_index, run_index)
        if annotation in _active.seen:
            raise ValueError(f"Annotation {annotation!r} was already used in this run")
        _active.seen.add(annotation)
        marker = f"{annotation} [nspy:{_active.token}:{len(_active.invocations)}]"
        _active.invocations[marker] = invocation
        return marker
