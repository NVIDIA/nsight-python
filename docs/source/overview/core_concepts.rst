.. SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Core Concepts
=============

Nsight Python requires two components working together to profile your GPU code:

- :func:`@nsight.analyze.kernel <nsight.analyze.kernel>` - A decorator on your benchmark function that runs the profiler and collects metrics
- :func:`nsight.annotate <nsight.annotate>` - A context manager (or decorator) used *inside* your benchmark function to mark which kernel(s) to measure

**Both are required.** The ``@nsight.analyze.kernel`` decorator controls the profiling session, while ``nsight.annotate`` tells the profiler which specific code regions to measure.

.. code-block:: python

   @nsight.analyze.kernel  # Controls profiling session
   def benchmark(n):
       a = torch.randn(n, n, device="cuda")
       b = torch.randn(n, n, device="cuda")

       with nsight.annotate("matmul"):  # Marks the kernel to measure
           c = a @ b

Nsight Python operates through three key primitives:

**1. Annotations**
An :func:`annotation <nsight.annotate>` wraps a region of code that launches a GPU kernel and tags it for profiling.
By default, each annotation should contain exactly one kernel launch. Annotations can be used as decorators or context managers:

.. code-block:: python

   @nsight.annotate("torch")
   def torch_kernel():
       ...

   # or
   with nsight.annotate("cutlass4"):
       cutlass_kernel()

If your annotated region launches multiple kernels, you have two options:

- Use ``replay_mode="range"`` to profile the entire annotated range as a unit, with metrics associated with the range rather than individual kernels
- Use ``combine_kernel_metrics`` to specify how to aggregate metrics across individual kernels (e.g., sum the runtimes)

See the :func:`@nsight.analyze.kernel <nsight.analyze.kernel>` documentation for details on these parameters.

**2. Kernel Analysis Decorator**  
Use :func:`nsight.analyze.kernel` to annotate a benchmark function. Nsight Python will rerun this function one configuration at a time. You can provide configurations in two ways:

- **At decoration time** using the `configs` parameter.
- **At function call time** by passing `configs` directly as an argument when invoking the decorated function.

.. code-block:: python

   @nsight.analyze.kernel
   def benchmark(s):
       ...

   benchmark(configs=[(1024,), (2048,)])

**3. Plot Decorator**  
Add :func:`nsight.analyze.plot` to automatically generate plots from your profiling runs.

.. code-block:: python

   @nsight.analyze.plot(filename="plot.png", ylabel="Runtime (ns)")
   @nsight.analyze.kernel(configs=[(1024,), (2048,)])
   def benchmark(s):
       ...
