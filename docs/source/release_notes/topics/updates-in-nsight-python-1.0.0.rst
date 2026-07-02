.. SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Updates in Nsight Python 1.0.0
==============================

Enhancements
------------

- **Eliminated per-decorator Python script relaunch**: Previously, nsight-python
  relaunched the entire Python script once for each :func:`@nsight.analyze.kernel <nsight.analyze.kernel>` decorated
  function in order to collect profiles using NVIDIA Nsight Compute. nsight-python now
  dynamically loads NVIDIA Nsight Compute's CUDA injection library at import time and uses
  ``ncu --mode attach`` to attach to the running Python process, with no script relaunch.
  This reduces profiling overhead and avoids relaunch-related side effects from
  re-running module/import-time code.
  Interactive environments such as Jupyter Notebook and Google Colab are now supported
  (fixes `#3 <https://github.com/NVIDIA/nsight-python/issues/3>`_ and `#19 <https://github.com/NVIDIA/nsight-python/issues/19>`_).

- **Added support for metric units**: The DataFrame returned by
  :meth:`~nsight.collection.core.ProfileResults.to_dataframe` now includes a ``Unit`` column. Units for
  metrics collected with NVIDIA Nsight Compute are parsed directly from the report. For derived
  metrics, ``derive_metric`` should return a ``(value, unit)`` tuple for a single
  metric, or a dictionary of ``(value, unit)`` tuples for multiple metrics. Returning
  only the value remains supported for backward compatibility, but emits a warning and
  sets ``Unit`` to ``np.nan``.

- **Improved decorated function parameter handling**
  (`#31 <https://github.com/NVIDIA/nsight-python/pull/31>`_):

  - **Keyword-only parameters** (defined after ``*`` in the signature) are now
    supported. Config values are mapped correctly to both positional and keyword-only
    parameters.
  - **Default parameter values**: configs can now omit trailing parameters that have
    defaults — missing values are filled in automatically from the function signature.
  - ``*args`` / ``**kwargs`` no longer cause false validation errors; they are
    excluded from parameter counting and ignored during profiling.

- **Added cuTile kernel example**
  (`#37 <https://github.com/NVIDIA/nsight-python/pull/37>`_):
  Added a new example demonstrating how to profile a CUDA Tile kernel.

- **Added thermal device selection**
  (`#51 <https://github.com/NVIDIA/nsight-python/pull/51>`_, fixes
  `#35 <https://github.com/NVIDIA/nsight-python/issues/35>`_):
  Added a ``thermal_device`` parameter to
  :func:`@nsight.analyze.kernel <nsight.analyze.kernel>` for pinning
  Thermovision's thermal monitoring to a specific CUDA device ordinal. If
  unset, Thermovision now maps the current CUDA device context to its
  underlying NVML device by UUID (honoring ``CUDA_VISIBLE_DEVICES``) instead
  of always monitoring physical GPU 0, and tracks CUDA context switches (for
  example via ``torch.cuda.set_device``) made during profiling.

API Changes
-----------

- **Consistent output prefix**: Progress bar, config, and header output
  printed by nsight-python is now consistently prefixed with
  ``[NSIGHT-PYTHON]``.

- **Replaced** ``output`` **with** ``verbosity``: The ``output`` parameter of
  :func:`@nsight.analyze.kernel <nsight.analyze.kernel>`, which accepted the strings
  ``"quiet"``, ``"progress"``, and ``"verbose"``, has been replaced by a
  ``verbosity`` parameter of type :class:`~nsight.utils.VerbosityLevel`:

  - ``VerbosityLevel.SILENT`` (replaces ``"quiet"``): suppresses all output.
  - ``VerbosityLevel.INFO`` (replaces ``"progress"``): shows the progress bar,
    profiling completion messages, and report file paths. This is the default.
  - ``VerbosityLevel.DEBUG`` (replaces ``"verbose"``): additionally prints the
    full NCU CLI command and enables NCU verbose output.

Fixes
-----

- **Fixed crashes with tuple-valued function arguments**
  (`#30 <https://github.com/NVIDIA/nsight-python/pull/30>`_):
  Fixed crashes and corrupted extraction results when profiled functions have
  tuple-valued arguments. Tuple arguments are now preserved correctly for
  single- and multi-metric profiling.

- **Fixed crashes with unhashable config parameters**
  (`#42 <https://github.com/NVIDIA/nsight-python/pull/42>`_):
  Fixed crashes when aggregating profiling results for configurations containing
  dictionary- or list-valued arguments. These configuration parameters are now
  handled consistently regardless of the number of runs.

- **Fixed misleading y-axis labels on non-normalized plots**
  (`#33 <https://github.com/NVIDIA/nsight-python/pull/33>`_):
  Plots of non-normalized data no longer show a spurious "relative to False"
  suffix on the y-axis label.
