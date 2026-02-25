.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Updates in Nsight Python 0.9.6
==============================

Fixes
-----

- `Github Issue #13 <https://github.com/NVIDIA/nsight-python/issues/13>`_:
  Fixed incorrect profiling results when making multiple function calls of the same decorated function.

- `Github Issue #17 <https://github.com/NVIDIA/nsight-python/issues/17>`_:
  Fixed ``ZeroDivisionError`` when handling zero-valued metrics (e.g., ``sm__idc_divergent_instructions.avg``).

Enhancements
------------

- `Github Issue #11 <https://github.com/NVIDIA/nsight-python/issues/11>`_:
  Added support for **multiple derived metrics** using ``derive_metric`` parameter. The ``derive_metric`` function can now return either a single value or a dictionary of multiple metrics.

- Added **metric parameter** to ``@nsight.analyze.plot`` decorator to specify which metric to visualize when multiple metrics are collected.

- **Normalization improvements**:

  - Changed ``normalize_against`` to use standard normalization (current/baseline) instead of appending normalization info to metric names.
  - Added **Normalized** column to the output dataframe to indicate which annotation is used for normalization.

- **Adaptive Thermovision**: Replaced ``thermal_control`` boolean parameter with ``thermal_mode`` parameter that accepts ``"auto"``, ``"manual"``, or ``"off"`` values for more flexible thermal throttling control.

Other Changes
-------------

- Improved documentation clarity and fixed gaps in user-facing documentation.
- Added test coverage for zero-valued metrics to ensure proper handling.
