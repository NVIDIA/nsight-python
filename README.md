[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Documentation](https://img.shields.io/badge/Nsight%20Python-documentation-brightgreen.svg)](https://docs.nvidia.com/nsight-python/)

# Nsight Python

Nsight Python is a Python kernel profiling interface based on NVIDIA Nsight Tools.
It simplifies performance benchmarking and visualization of performance metrics — all in just a few lines of Python.

Nsight Python helps you unlock peak performance from your GPU kernels by simplifying performance benchmarking and visualization — all in just a few lines of Python code.

## Installation

Please refer to the [Installation documentation](https://docs.nvidia.com/nsight-python/installation/runtime_requirements.html) for detailed instructions.

## Installation from source

Install as an editable install:

```bash
pip install -e .
```

If you want to manage all run-time dependencies yourself, also pass the `--no-deps` flag.

## Running tests

Tests require NVIDIA Nsight Compute to be installed and available in your PATH.

### Prerequisites

Install pytest:

```bash
pip install pytest
```

### PyTorch Dependency

Most tests and examples require PyTorch for GPU operations:

```bash
# Install PyTorch with CUDA support matching your system (e.g., CUDA 12.6, 12.9, 13.0)
# Replace cuXXX with your CUDA version (e.g., cu126, cu129, cu130)
pip install torch --index-url https://download.pytorch.org/whl/cuXXX
```

Visit [pytorch.org](https://pytorch.org/get-started/locally/) for installation commands matching your specific CUDA version.

### Running Tests

```bash
pytest tests -v       # Run just unit tests
pytest examples -v    # Run just the examples
pytest -v             # Run the tests and examples
```

## Contributing Guide

Review the [CONTRIBUTING.md](Contributing.md) file for information on how to contribute code and issues to the project.

## License

All files hosted in this repository are subject to the [Apache 2.0](LICENSE) license.

## Disclaimer

nsight-python is in a Beta state. Beta products may not be fully functional, may contain errors or design flaws, and may be changed at any time without notice. We appreciate your feedback to improve and iterate on our Beta products.
