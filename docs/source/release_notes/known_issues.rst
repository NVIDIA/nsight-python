.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Known Issues
============

- Nsight Python launches the application python script again with NVIDIA Nsight Compute CLI (ncu) for each ``@nsight.analyze.kernel`` decorator.

  - Due to this issue, using ``nsight-python`` in interactive notebook environments
    (e.g., Jupyter Notebook or Google Colab) is not supported. Refer to
    `GitHub issue #3 <https://github.com/NVIDIA/nsight-python/issues/3>`_
    for more information..

  In the meantime, you can use the following workaround by placing your benchmark
  into a standalone Python script and executing it outside the interactive cell environment::

      %%writefile quickstart.py
      import torch
      import nsight

      @nsight.analyze.kernel
      def benchmark_matmul(n):
          """
          The simplest possible benchmark.
          We create two matrices and multiply them.
          """
          # Create two NxN matrices on GPU
          a = torch.randn(n, n, device="cuda")
          b = torch.randn(n, n, device="cuda")

          # Mark the kernel we want to profile
          with nsight.annotate("matmul"):
              c = a @ b

          return c

      result = benchmark_matmul(1024)

  Then run the script with::

      %run quickstart.py

- Kernels launched from a subprocess which is created within the annotated region will not be profiled.
- For the ``nsight.analyze.kernel``'s ``replay_mode="range"`` option, only a subset of CUDA APIs are supported within the annotated range. If an unsupported API call is detected, an error will be reported. For details on supported APIs, refer to the `NVIDIA Nsight Compute Profiling Guide <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#supported-apis>`_. In such cases, you can either switch to ``replay_mode="kernel"`` or modify the code to exclude the unsupported API from the annotated range.
- Nested annotations (using ``nsight.annotate`` within another ``nsight.annotate`` context) are not supported. nsight-python errors out when nested annotations are used.
- ``*args`` and ``**kwargs`` in decorated function signatures are tolerated but ignored — they will always be empty and will not appear in the profiling output.
