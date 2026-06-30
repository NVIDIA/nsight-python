# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import version

from nsight import analyze
from nsight.annotation import annotate
from nsight.collection.ncu import try_init_injection
from nsight.utils import VerbosityLevel, col_panel, row_panel

# Load injection library and symbols for NCU attach (if available)
try_init_injection()

__version__ = version("nsight-python")

__all__ = ["analyze", "annotate", "VerbosityLevel"]
