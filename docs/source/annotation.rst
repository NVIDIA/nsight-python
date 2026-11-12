.. SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

nsight.annotate
================

.. note::
   ``nsight.annotate`` marks NVTX regions but does not select or run a tool.
   To collect profiling metrics, use it inside a function decorated with
   :func:`@nsight.analyze.kernel <nsight.analyze.kernel>`; see
   :doc:`/overview/core_concepts`. For Compute Sanitizer, activate
   :attr:`nsight.Tool.COMPUTE_SANITIZER` and use ``nsight.annotate`` without
   that decorator; see :ref:`tools-compute-sanitizer`.

.. autoclass:: nsight.annotate
