.. SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Installing from PyPI
====================

Nsight Python can be installed directly from the Python Package Index (PyPI).

Basic Installation
------------------

For most users, a basic installation is sufficient:

.. code-block:: bash

    pip install nsight-python

This installation works with both CUDA Toolkit 12 and CUDA Toolkit 13, and provides all core Nsight Python features.

Optional: Installing with cuda-core Support
--------------------------------------------

If you want to use the ``ignore_failures`` feature in ``nsight.annotate``, you need to install the ``cuda-core`` package. 
This is an optional dependency that enables enhanced error handling within annotated regions.

For CUDA Toolkit 12:

.. code-block:: bash

    pip install nsight-python[cu12]

For CUDA Toolkit 13:

.. code-block:: bash

    pip install nsight-python[cu13]

Optional: Installing with CUPTI Support
----------------------------------------

If you want to use the experimental CUPTI backend with ``nsight.analyze.kernel``, you need to install the ``cupti-python`` package.
This is an optional dependency that enables low-overhead collection of kernel time durations.

.. code-block:: bash

    pip install nsight-python[cupti]
