# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import version

from nsight import analyze
from nsight.annotation import annotate
from nsight.utils import col_panel, row_panel

__version__ = version("nsight-python")

__all__ = ["analyze", "annotate"]
