# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import enum
import warnings

from nsight.collection import cupti, ncu


class Tool(str, enum.Enum):
    """The profiling/analysis tools Nsight Python supports."""

    NCU = "ncu"
    CUPTI = "cupti"


_active_tool: Tool | None = None


def activate(tool: Tool) -> None:
    """Select and load ``tool`` as the active tool for this process.

    Only one tool can be active at a time. The first call loads the tool; calling
    again with the *same* tool is a no-op. Calling with a *different* tool while
    one is already active raises, because the tools are mutually exclusive.

    Each tool has its own prerequisites and its own constraints on when it may
    be activated. See the :doc:`/tools` page for the tool you are selecting.

    Args:
        tool: The tool to activate.

    Raises:
        ValueError: If ``tool`` is not a :class:`Tool`.
        RuntimeError: If a different, incompatible tool is already active.
        ProfilerException: If the selected tool fails to load (wrong version,
            injection library missing, etc.). A tool whose prerequisites are
            missing may raise its own exception type instead.
    """
    global _active_tool
    if not isinstance(tool, Tool):
        raise ValueError(
            f"tool must be a nsight.Tool member (one of {[t.name for t in Tool]}), "
            f"got {tool!r}"
        )
    if _active_tool is not None:
        if _active_tool == tool:
            return
        raise RuntimeError(
            f"{_active_tool!r} is already active; cannot switch to {tool!r}"
        )
    if tool == Tool.NCU:
        ncu.init_injection()
    elif tool == Tool.CUPTI:
        cupti.init()
    _active_tool = tool


def deactivate() -> None:
    """Stop the currently active tool, if any.

    Stops and tears down the tool :func:`activate` last activated. It is a
    no-op if no tool is active.

    Not every tool can be torn down. A tool that cannot be stopped stays loaded
    for the lifetime of the process, so it also stays the active tool and this
    call is a no-op that warns rather than clearing the selection; no other tool
    can be activated afterwards. See the :doc:`/tools` page for which tools this
    applies to.

    Raises:
        ProfilerException: If tearing down the active tool fails.
    """
    global _active_tool
    if _active_tool == Tool.NCU:
        warnings.warn(
            "NVIDIA Nsight Compute cannot be deactivated once loaded; it stays "
            "the active tool for the lifetime of the process.",
            category=RuntimeWarning,
            stacklevel=2,
        )
    elif _active_tool == Tool.CUPTI:
        cupti.finalize()
        _active_tool = None


def get_active_tool() -> Tool | None:
    """Return the tool currently activated by :func:`activate`, or ``None``.

    Returns ``None`` if no tool is active (``activate()`` has not been called, or
    the active tool was cleared by :func:`deactivate`). Doubles as an "is anything
    active?" check and lets callers branch on the active tool -- e.g. skipping a
    test that needs a metric only one tool exposes.

    Returns:
        The active :class:`Tool`, or ``None`` if no tool is active.
    """
    return _active_tool
