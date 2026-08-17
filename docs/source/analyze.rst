.. SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

nsight.analyze
===============

.. note::
   ``@nsight.analyze.kernel`` runs the profiling session, but you must use :func:`nsight.annotate <nsight.annotate>` inside
   your decorated function to mark which kernel(s) to measure. See :doc:`/overview/core_concepts` for details.

   The decorator returns a :class:`~nsight.collection.core.ProfileResults` object containing the collected metrics.
   See :doc:`/collection/core` for full API documentation.

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
