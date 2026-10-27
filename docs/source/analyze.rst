.. SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

nsight.analyze
===============

.. note::
   ``@nsight.analyze.kernel`` runs the profiling session, but you must use :func:`nsight.annotate <nsight.annotate>` inside
   your decorated function to mark which kernel(s) to measure. See :doc:`/overview/core_concepts` for details.

   The decorator returns a :class:`~nsight.collection.core.ProfileResults` object containing the collected metrics.
   See :doc:`/overview/profiling_results` for the result columns and
   :doc:`/collection/core` for full API documentation.

.. warning::
   Decorating a function with ``@nsight.analyze.kernel`` changes its return value.
   Any value returned by the original function is ignored, and calling the
   decorated function returns a :class:`~nsight.collection.core.ProfileResults`
   object instead. If the original function returns a non-``None`` value,
   Nsight Python emits a ``RuntimeWarning``.

   If you also need the computational result, keep that computation in an
   undecorated helper and use a decorated wrapper only for profiling. The
   decorated wrapper should not return the helper's result:

   .. code-block:: python

      import torch

      import nsight

      def compute(n):
          return torch.ones(n, device="cuda")

      @nsight.analyze.kernel
      def profile_compute(n):
          with nsight.annotate("compute"):
              compute(n)

      output = compute(1024)
      profile_results = profile_compute(1024)

.. autoclass:: nsight.analyze.kernel

.. autoclass:: nsight.analyze.plot

.. autoclass:: nsight.analyze.ignore_failures

Experimental information collectors
-----------------------------------

Custom information collectors add application-specific columns to profiling
results. The API is experimental and may change without a compatibility
guarantee. Use the ``scope`` parameter to select whether a collector runs once
per profiling session, once per configuration, once per repeated run, or once
per annotation.

.. autofunction:: nsight.experimental.collect

.. autoclass:: nsight.CollectionScope
   :no-undoc-members:

.. autoclass:: nsight.InfoCollector
   :no-undoc-members:

Variable kernel counts
----------------------

With ``combine_kernel_metrics``, each execution of an annotation is reduced
independently. For example, configurations launching two and three kernels can
be profiled in the same call, and counts may also vary between repeated runs.
The reduction preserves kernel launch order. Without a reducer, kernel replay
still requires one kernel per annotation execution.

Skipped annotations do not produce samples. An annotation that is entered but
contains no profiled kernels, including when all its kernels are filtered out,
raises an error identifying the configuration and run. Failed executions
suppressed with ``ignore_failures=True`` remain missing measurements.

Result tables retain the supplied annotation names. Raw NVTX labels in profiler
reports include a private execution suffix so results can be attributed without
assuming equal kernel counts. The processed ``Kernel`` column describes the
first sample's kernel sequence; inspect raw samples when sequences vary.
