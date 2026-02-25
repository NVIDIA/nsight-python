.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Updates in Nsight Python 0.9.6
==============================

Enhancements
------------

- `Github Issue #11 <https://github.com/NVIDIA/nsight-python/issues/11>`_:
  Added support for **multiple derived metrics** in :func:`nsight.analyze.kernel` using ``derive_metric`` parameter. The ``derive_metric`` function can now return either a single value or a dictionary of multiple metrics.

- Added **metric parameter** to :func:`nsight.analyze.plot` decorator to specify which metric to visualize when multiple metrics are collected.

- **Normalization improvements** in :func:`nsight.analyze.kernel`:

  - Changed ``normalize_against`` to use standard normalization (current/baseline) instead of appending normalization info to metric names.
  - Added **Normalized** column to the output dataframe to indicate which annotation is used for normalization.

- **Adaptive Thermovision** in :func:`nsight.analyze.kernel`:

  - Replaced ``thermal_control`` boolean parameter with ``thermal_mode`` parameter that accepts ``"auto"``, ``"manual"``, or ``"off"`` values for more flexible thermal throttling control.
  - Added ``thermal_wait`` parameter to specify the thermal headroom threshold (T.Limit in °C) that triggers cooling pause.
  - Added ``thermal_cont`` parameter to specify the thermal headroom threshold (T.Limit in °C) to resume profiling after cooling.
  - Added ``thermal_timeout`` parameter to specify the maximum wait time in seconds for GPU to cool down.

Fixes
-----

- `Github Issue #13 <https://github.com/NVIDIA/nsight-python/issues/13>`_:
  Fixed incorrect profiling results when making multiple function calls of the same decorated function.

- `Github Issue #17 <https://github.com/NVIDIA/nsight-python/issues/17>`_:
  Fixed ``ZeroDivisionError`` when handling zero-valued metrics (e.g., ``sm__idc_divergent_instructions.avg``).
  Added test coverage for zero-valued metrics to ensure proper handling.

