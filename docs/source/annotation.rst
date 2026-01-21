.. SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

nsight.annotate
================

.. note::
   ``nsight.annotate`` is used to mark which code regions to profile, but it does not run the profiler by itself.
   You must use it inside a function decorated with :func:`@nsight.analyze.kernel <nsight.analyze.kernel>` to actually collect metrics.
   See :doc:`/overview/core_concepts` for details.

.. autoclass:: nsight.annotate
