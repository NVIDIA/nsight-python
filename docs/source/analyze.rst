.. SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

nsight.analyze
===============

.. note::
   ``@nsight.analyze.kernel`` runs the profiling session, but you must use :func:`nsight.annotate <nsight.annotate>` inside
   your decorated function to mark which kernel(s) to measure. See :doc:`/overview/core_concepts` for details.

   The decorator returns a :class:`~nsight.collection.core.ProfileResults` object containing the collected metrics.
   See :doc:`/collection/core` for full API documentation.

.. autoclass:: nsight.analyze.kernel

.. autoclass:: nsight.analyze.plot

.. autoclass:: nsight.analyze.ignore_failures
