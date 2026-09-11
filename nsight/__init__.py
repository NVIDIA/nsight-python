# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from importlib.metadata import version

from nsight import analyze, exceptions, experimental
from nsight.annotation import annotate
from nsight.collection import ncu
from nsight.info_collector import CollectionScope, InfoCollector
from nsight.tools_manager import Tool, activate, deactivate, get_active_tool
from nsight.utils import VerbosityLevel, col_panel, row_panel

# NVIDIA Nsight Compute must attach before CUDA is initialized, so by default it
# is activated here -- importing nsight early is enough to satisfy that. Set
# NSPY_NCU_INIT_AT_IMPORT=0 to disable this and activate a tool yourself.
if os.environ.get("NSPY_NCU_INIT_AT_IMPORT", "1") != "0":
    try:
        activate(Tool.NCU)
    except (exceptions.ProfilerException, exceptions.NCUNotAvailableError) as exc:
        # Importing nsight must keep working when the NVIDIA Nsight Compute
        # injection fails to load, so record the error rather than raising it
        # here. NCUCollector.collect re-raises it late, when a decorated
        # function is first called.
        ncu.injection_load_error = exc

__version__ = version("nsight-python")

__all__ = [
    "CollectionScope",
    "InfoCollector",
    "Tool",
    "VerbosityLevel",
    "activate",
    "analyze",
    "annotate",
    "deactivate",
    "experimental",
    "get_active_tool",
]
