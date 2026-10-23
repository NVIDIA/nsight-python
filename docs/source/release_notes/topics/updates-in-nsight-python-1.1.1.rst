.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Updates in Nsight Python 1.1.1
==============================

API Changes
-----------

- **Raise an error when no kernels were profiled**
  (`#83 <https://github.com/NVIDIA/nsight-python/pull/83>`_):
  :func:`@nsight.analyze.kernel <nsight.analyze.kernel>` now raises a
  ``RuntimeError`` when no kernel launch could be attributed to any
  :func:`nsight.annotate` region, instead of returning an empty DataFrame.
  Previously, with the CUPTI backend this failed silently when CUDA was used
  before the first profiled call.

- **Warn on ignored annotation failures**
  (`#83 <https://github.com/NVIDIA/nsight-python/pull/83>`_):
  When an annotation created with ``ignore_failures=True`` raises, a
  ``RuntimeWarning`` naming the annotation and the ignored exception is now
  emitted. The measured metric for that run is still set to ``NaN``.

Fixes
-----

- **Custom info is no longer read back from the output directory**:
  Values gathered by ``info_collectors`` are now passed from collection to
  extraction in memory. Previously they were saved to ``collected_info.pkl``
  under ``output_prefix`` and loaded from that path, so a file already at that
  path could be loaded in place of the collected values. ``collected_info.pkl``
  is no longer written, and collector values that cannot be pickled are no
  longer replaced with ``None``.

- **Improved kernel-count and missing-report errors**
  (`#83 <https://github.com/NVIDIA/nsight-python/pull/83>`_):
  The errors raised when an annotation launches more than one kernel, or a
  different number of kernels in different runs, now report the kernel and run
  counts and suggest warming up code that autotunes on its first call. The
  error raised when no NVIDIA Nsight Compute report is found now includes the
  report path and the most likely cause.

- **Plain-text output when stdout is not a terminal**
  (`#83 <https://github.com/NVIDIA/nsight-python/pull/83>`_):
  ANSI color and cursor-movement codes are only emitted when stdout is a
  terminal. When output is redirected to a file, a pipe, or a captured log,
  only the final progress bar is printed.
