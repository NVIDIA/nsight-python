.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Updates in Nsight Python 1.1.0
==============================

Enhancements
------------

- **Added explicit tool management APIs**:
  Added ``nsight.Tool``, :func:`nsight.activate`, :func:`nsight.deactivate`,
  and :func:`nsight.get_active_tool` for profiling tool life cycle management.
  See :doc:`/tools`.

- **Added experimental CUPTI backend**:
  Kernel durations can now be collected with CUPTI, a lightweight alternative
  to NVIDIA Nsight Compute. To get started with the CUPTI backend in
  :func:`@nsight.analyze.kernel <nsight.analyze.kernel>`, see
  :ref:`tools-cupti`. For installation and runtime requirements, please see
  :doc:`Installation </installation/installation_from_pypi>` and
  :doc:`Runtime Requirements </installation/runtime_requirements>`.

- **Added experimental custom information collectors**: Profiling results can
  now include application-specific information collected once per session,
  once per configuration, once per repeated run, or once per annotation. Use
  :func:`nsight.experimental.collect` with a
  :class:`~nsight.info_collector.CollectionScope` and pass the decorated
  functions through the ``info_collectors`` parameter of
  :func:`@nsight.analyze.kernel <nsight.analyze.kernel>`.

- **Added support for kernels launched from other threads**:
  Kernels launched from a thread other than the one that creates the annotated
  region are now profiled. :func:`nsight.annotate` now marks the region with an
  NVTX start/end range instead of a thread-local push/pop range, so NVIDIA
  Nsight Compute targets kernels launched from any thread within the region.

- **Added Windows compatibility**
  (fixes `#58 <https://github.com/NVIDIA/nsight-python/issues/58>`_).

Fixes
-----

- **Fixed aggregation failures with tensor-valued config parameters**
  (`#63 <https://github.com/NVIDIA/nsight-python/pull/63>`_).
