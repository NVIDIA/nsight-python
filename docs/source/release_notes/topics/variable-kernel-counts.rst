.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Variable kernel counts (unreleased)
===================================

Each annotation execution is now attributed to its configuration and run.
With ``combine_kernel_metrics``, the number of kernels may differ across
configurations and repeated runs. Empty or fully filtered executions produce
missing samples with a warning, retaining the remaining results.

Raw profiler reports contain private execution labels. Extraction helpers now
require the matching in-memory ``invocations`` mapping as a keyword argument;
a saved ``.ncu-rep`` alone is no longer sufficient for standalone extraction.
Returned result objects and exported CSVs remain available for later analysis.
