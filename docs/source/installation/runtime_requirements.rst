.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Runtime Requirements
=====================

* **NVIDIA Nsight Compute from CUDA Toolkit 13.3 Update 1 or later**: nsight-python uses the NVIDIA Nsight Compute CLI (ncu 2026.2.1+). Ensure ``ncu`` is installed and available on the system path.

Import Ordering
---------------

``import nsight`` must come before:

* **CUDA initialization**: You must import nsight before any operation that initializes CUDA (e.g. ``torch.cuda.init()``). If CUDA is initialized first, NVIDIA Nsight Compute cannot attach correctly and profiling fails (commonly with ``NV_INJ_ERROR_NOT_INITIALIZED``).

  **Recommendation**: Place ``import nsight`` at the top of your script, before all CUDA-related imports and calls.

* **Any NVTX API calls**: nsight-python updates ``NVTX_INJECTION64_PATH`` at import time. Any NVTX usage before ``import nsight`` will cause NVTX annotation ranges to be missed and profiling results to be incomplete.
