.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Architecture
============

Nsight Python's architecture consists of:

1. **Collection**: Runs your benchmark under NVIDIA Nsight Compute. See :doc:`../collection/index`.
2. **Extraction**: Parses Nsight reports using `ncu-report` and associates metrics with annotations/configs. See :doc:`../extraction`.
3. **Visualization**: Converts data to a pandas DataFrame and optionally plots results via matplotlib. See :doc:`../visualization`.

Internally, Nsight Python:

- Inserts NVTX ranges for each annotation
- Profiles each configuration for multiple runs
- Associates collected metrics with annotations
- Supports TFLOPs, speedup, and other derived metrics

Advanced Options
----------------

**Metric Selection**  
Nsight Python collects `gpu__time_duration.sum` by default. To collect other NVIDIA Nsight Compute metrics:

.. code-block:: python

   @nsight.analyze.kernel(metrics=["sm__throughput.avg.pct_of_peak_sustained_elapsed"])
   def benchmark1(...):
       ...

   # or
   @nsight.analyze.kernel(
       metrics=[
           "smsp__sass_inst_executed_op_shared_ld.sum",
           "smsp__sass_inst_executed_op_shared_st.sum",
           "launch__sm_count",
       ],
   )
   def benchmark2(...):
       ...

**Derived Metrics**  
Define a Python function that computes metrics like TFLOPs based on runtime and input configuration:

.. code-block:: python

   def tflops(t, m, n, k):
       return 2 * m * n * k / (t / 1e9) / 1e12

   @nsight.analyze.kernel(configs=[(1024, 1024, 64)], derive_metric=tflops)
   def benchmark(m, n, k):
       ...

**Relative Metrics**  
Compare performance against a baseline annotation:

.. code-block:: python

   @nsight.analyze.kernel(normalize_against="torch.einsum")
   def benchmark(...):
       ...

**Multiple Annotations**
Profile multiple implementations side-by-side:

.. code-block:: python

   with nsight.annotate("torch"):
       torch_impl(...)

   with nsight.annotate("cutlass4"):
       cutlass_impl(...)

**Multiple Config Parameters**

Nsight Python supports multi-dimensional config tuples which can contain arbitrary Python objects:

.. code-block:: python

   import itertools
   configs = list(itertools.product([512, 1024], [64, 128]))  # (seqlen, head_dim)

   @nsight.analyze.kernel(configs=configs)
   def benchmark(seqlen, head_dim):
       ...
