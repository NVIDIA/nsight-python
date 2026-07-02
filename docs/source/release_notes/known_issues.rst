.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Known Issues
============

- Kernels launched from a subprocess which is created within the annotated region will not be profiled.
- For the ``nsight.analyze.kernel``'s ``replay_mode="range"`` option, only a subset of CUDA APIs are supported within the annotated range. If an unsupported API call is detected, an error will be reported. For details on supported APIs, refer to the `NVIDIA Nsight Compute Profiling Guide <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#supported-apis>`_. In such cases, you can either switch to ``replay_mode="kernel"`` or modify the code to exclude the unsupported API from the annotated range.
- Nested annotations (using ``nsight.annotate`` within another ``nsight.annotate`` context) are not supported. nsight-python errors out when nested annotations are used.
- ``*args`` and ``**kwargs`` in decorated function signatures are tolerated but ignored — they will always be empty and will not appear in the profiling output.
- Calling ``nsight.analyze.kernel``-decorated functions from multiple threads is not supported. This includes running two decorated calls concurrently, as in ``examples/00_minimal.py``:

  .. code-block:: python

      @nsight.analyze.kernel
      def benchmark_matmul(n: int) -> torch.Tensor:
          a = torch.randn(n, n, device="cuda")
          b = torch.randn(n, n, device="cuda")

          with nsight.annotate("matmul"):
              c = a @ b

      matmul_1 = threading.Thread(target=benchmark_matmul, args=(1024,))
      matmul_2 = threading.Thread(target=benchmark_matmul, args=(2048,))

      matmul_1.start()
      matmul_2.start()

      matmul_1.join()
      matmul_2.join()

  It also includes running the decorated calls one after another on separate
  threads, with each thread fully joined before the next is started:

  .. code-block:: python

      matmul_1 = threading.Thread(target=benchmark_matmul, args=(1024,))
      matmul_1.start()
      matmul_1.join()

      matmul_2 = threading.Thread(target=benchmark_matmul, args=(2048,))
      matmul_2.start()
      matmul_2.join()

  Neither pattern is supported — profiling must be driven from a single thread.
