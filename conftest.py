# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

# How each --tool-init mode sets nsight up, as
# (NSPY_NCU_INIT_AT_IMPORT, tool to activate in pytest_configure).
_INIT_MODES = {
    # nsight activates NCU itself while `import nsight` runs.
    "ncu-at-import": ("1", None),
    # Nothing is active up front, so nsight.analyze.kernel activates NCU on the
    # first call to a decorated function.
    "ncu-lazy": ("0", None),
    # CUPTI is activated here instead; nsight must not claim NCU at import.
    "cupti": ("0", "cupti"),
}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--tool-init",
        default="ncu-at-import",
        choices=sorted(_INIT_MODES),
        help=(
            "How the nsight tool is initialized for the suite. 'ncu-at-import' "
            "(the default) lets nsight activate NCU while it is imported. "
            "'ncu-lazy' leaves no tool active so the lazy NCU load in "
            "nsight.analyze.kernel is exercised on the first decorated call. "
            "'cupti' activates CUPTI before the suite runs. Only one tool can be "
            "active per process, so this mode fully determines both "
            "NSPY_NCU_INIT_AT_IMPORT and which tool is activated."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    init_at_import, tool_name = _INIT_MODES[config.getoption("--tool-init")]

    # Set before nsight is imported: it reads this while importing, which is why
    # the import below is here rather than at module scope. pytest_configure
    # still runs before any test module is imported, so anything activated here
    # is ahead of CUDA initialization.
    os.environ["NSPY_NCU_INIT_AT_IMPORT"] = init_at_import

    import nsight

    if tool_name is not None:
        nsight.activate(nsight.Tool(tool_name))
