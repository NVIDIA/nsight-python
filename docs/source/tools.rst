.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Tools
=====

Nsight Python collects its data through an underlying NVIDIA tool. Only one tool
can be active at a time, and :func:`nsight.activate` selects and loads it.

.. autofunction:: nsight.activate

.. autofunction:: nsight.deactivate

.. autofunction:: nsight.get_active_tool

.. autoclass:: nsight.Tool

Supported tools
---------------

NVIDIA Nsight Compute
^^^^^^^^^^^^^^^^^^^^^

:attr:`nsight.Tool.NCU` profiles individual kernels with the NVIDIA Nsight
Compute CLI. It is the default tool for ``@nsight.analyze.kernel``.

Activation
""""""""""

NVIDIA Nsight Compute's injection library has to be loaded before CUDA is
initialized. It is loaded in one of three ways.

* **At import, by default.** ``import nsight`` activates NCU.
* **Explicitly, for control over the timing.** Ensure the environment variable
  ``NSPY_NCU_INIT_AT_IMPORT`` is set to ``0`` before ``nsight`` is imported,
  then activate NCU where it suits you. Explicit activation also surfaces setup
  errors at that point.

  .. code-block:: python

     import os

     os.environ["NSPY_NCU_INIT_AT_IMPORT"] = "0"

     import nsight

     nsight.activate(nsight.Tool.NCU)

* **Lazily, on first use.** Under ``NSPY_NCU_INIT_AT_IMPORT=0``, the first call
  to a function decorated with ``@nsight.analyze.kernel`` loads the library.
  Ensure the first decorated call comes before any code that initializes CUDA.

Limitations
"""""""""""

**Deactivation.** :func:`~nsight.deactivate` is not supported. The injection
library cannot be unloaded, so NCU stays loaded for the lifetime of the process
and remains the active tool once activated.

.. _tools-cupti:

NVIDIA CUPTI (experimental)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

:attr:`nsight.Tool.CUPTI` is an opt-in, low-overhead alternative for collecting
kernel time durations, measured using CUPTI's Activity API.

Activation
""""""""""

To activate CUPTI, ensure the environment variable
``NSPY_NCU_INIT_AT_IMPORT`` is set to ``0`` before ``nsight`` is imported:

.. code-block:: python

   import os

   os.environ["NSPY_NCU_INIT_AT_IMPORT"] = "0"

   import torch

   import nsight

   nsight.activate(nsight.Tool.CUPTI)


   @nsight.analyze.kernel
   def benchmark_matmul(n):
       a = torch.randn(n, n, device="cuda")
       b = torch.randn(n, n, device="cuda")

       with nsight.annotate("matmul"):
           c = a @ b


   if __name__ == "__main__":
       result = benchmark_matmul(1024)
       print(result.to_dataframe())

       nsight.deactivate()

Activating CUPTI updates ``NVTX_INJECTION64_PATH``. Any NVTX usage before
``nsight.activate(nsight.Tool.CUPTI)`` will cause NVTX annotation ranges to be
missed and profiling results to be incomplete.

CUPTI may be activated either before or after CUDA is initialized.

Limitations
"""""""""""

**Platform.** Linux only for now. The CUPTI backend is not supported on Windows.

**Metrics.** Only ``"gpu__time_duration.sum"`` is supported. CUPTI measures it
by recording kernel start and end timestamps from CUPTI activity records
(``CONCURRENT_KERNEL``) and computing ``end - start`` per kernel launch.

**Replay mode.** Neither kernel replay nor range replay is supported. CUPTI
activity records capture one execution of each kernel in a single pass, so no
replay mechanism is engaged and ``replay_mode`` is ignored.

**Clock and cache control.** ``clock_control`` and ``cache_control`` are
ignored. Clocks are not locked and caches are not flushed between runs, which
may affect measurement stability.

**Device synchronization.** The whole device is synchronized once after the
profiling session, before the activity buffers are flushed, because a kernel's
activity record is only written when the kernel completes. Work that the
profiled function left running asynchronously is therefore waited for before
profiling returns.

.. _tools-compute-sanitizer:

NVIDIA Compute Sanitizer (experimental)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:attr:`nsight.Tool.COMPUTE_SANITIZER` checks CUDA work in annotated regions.

Activation
""""""""""

Set ``NSPY_NCU_INIT_AT_IMPORT=0`` before importing ``nsight`` to prevent
default NCU activation. Activate Compute Sanitizer before importing a CUDA
framework or otherwise initializing CUDA. Activation self-injects the
collection backend into the current process and starts one attach frontend;
the Python script is not relaunched.

The executable on ``PATH`` may be the CUDA Toolkit's
``bin/compute-sanitizer`` launcher. Nsight Python discovers its matching
``libsanitizer-collection.so`` and ``libsanitizer-public.so`` in the sibling
``compute-sanitizer`` directory, including through a toolkit symlink such as
``/usr/local/cuda``.

The following complete example uses ``memcheck`` to detect an out-of-bounds
CUDA memset. Run it with ``python examples/14_compute_sanitizer.py`` from the
repository root.

.. literalinclude:: ../../examples/14_compute_sanitizer.py
   :language: python

For a workload with shared-memory races, replace the example's activation
options with:

.. code-block:: python

   nsight.ComputeSanitizerOptions(
       tool=nsight.ComputeSanitizerTool.RACECHECK,
       tool_args=("--racecheck-report", "all"),
   )

The example's invalid memset is a ``memcheck`` case, not a racecheck case.

``memcheck`` is the default tool. ``tool_args`` passes individual CLI argument
tokens to the analyzer; it is not a shell command string. Nsight Python manages
the connection, NVTX filters, and text log output itself.

.. autoclass:: nsight.ComputeSanitizerTool

.. autoclass:: nsight.ComputeSanitizerOptions

The text log is updated while the process runs. ``log_path`` is optional;
without it, Nsight Python selects a unique ``.log`` filename in the system
temporary directory. An explicit path must not already exist, and its parent
directory must exist. Use ``nsight.sanitize.log_path()`` to get the selected
destination. The Compute Sanitizer frontend may not have created the file yet,
so check ``Path.exists()`` before reading it.

Limitations
"""""""""""

**Platform.** Self-attach currently requires Linux with ``/proc``.

**Deactivation.** :func:`~nsight.deactivate` is not supported. The injected
backend cannot be unloaded, so Compute Sanitizer stays loaded for the lifetime
of the process and remains the active tool once activated.
