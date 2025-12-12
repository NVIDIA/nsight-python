# Nsight Python Examples

This directory contains examples demonstrating how to use Nsight Python for profiling and visualizing CUDA kernel performance.

## Prerequisites

### Required

- **Python 3.10+**
- **CUDA-capable GPU**
- **NVIDIA Nsight Compute** (for profiling)

### Python Dependencies

The examples require additional packages beyond the base `nsight` package:

#### PyTorch

Most examples use PyTorch for GPU operations:

```bash
# Install PyTorch with CUDA support matching your system (e.g., CUDA 12.6, 12.9, 13.0)
# Replace cuXXX with your CUDA version (e.g., cu126, cu129, cu130)
pip install torch --index-url https://download.pytorch.org/whl/cuXXX
```

Visit [pytorch.org](https://pytorch.org/get-started/locally/) for installation commands matching your specific CUDA version.

#### Triton (Optional)

For the Triton examples (`07_triton_minimal.py`):

```bash
pip install triton
```

## Quick Start

The examples are numbered in order of complexity. Start with `00_minimal.py`:

```bash
cd examples
python 00_minimal.py
```

This will profile a simple matrix multiplication and generate a plot showing the performance.

## Examples Overview

- **`00_minimal.py`** - Simplest possible benchmark
  - Basic `@nsight.analyze.kernel` usage
  - Single parameter configuration sweep
  - Default time-based profiling

- **`01_compare_throughput.py`** - Comparing implementations
  - Multiple annotated regions (different matmul implementations)
  - Using NSight Compute metrics (DRAM throughput)
  - Using `@nsight.annotate` as a function decorator
  - Printing collected data with `print_data=True`

- **`02_parameter_sweep.py`** - Sweeping parameters
  - Multiple configuration values
  - Visualizing performance across problem sizes

- **`03_custom_metrics.py`** - Computing TFLOPs
  - Using `derive_metric` to compute custom metric
  - Understanding the metric function signature
  - Transforming time measurements into performance metrics

- **`04_multi_parameter.py`** - Multiple parameters
  - Using `itertools.product()` for parameter combinations
  - Flexible metric functions with `*conf` pattern
  - Handling multiple configuration dimensions

- **`05_subplots.py`** - Creating subplot grids
  - Using `row_panels` and `col_panels`
  - Organizing multi-dimensional data visually
  - Creating publication-ready plots

- **`06_plot_customization.py`** - Advanced plotting
  - Customizing plot appearance
  - Using `plot_callback` for advanced control
  - Line plots vs bar charts
  - Annotating data points

- **`07_triton_minimal.py`** - Profiling Triton kernels
  - Integrating Triton GPU kernels
  - Using `variant_fields` and `variant_annotations`
  - Comparing against PyTorch baselines with `normalize_against`
  - Showing speedup metrics

- **`08_multiple_metrics.py`** - Collecting multiple metrics
  - Collecting multiple metrics with using a sequence of metric names
  - Merged results with `"Metric"` column in DataFrame
  - `@plot` decorator incompatible with multiple metrics

- **`09_advanced_metric_custom.py`** - Computing advanced custom metric
  - Using `derive_metric` to compute custom metric from multiple metrics

- **`10_multiple_kernels_combine.py`** - Combining metrics from multiple kernels
  - Using `combine_kernel_metrics` to aggregate metrics from multiple kernels
  - Summing metrics from consecutive kernel executions

- **`11_output_csv.py`** - Outputting to CSV
  - Using `output_csv` parameter to enable/disable CSV file generation
  - Using `output_prefix` to specify output file location and naming
  - Reading and displaying generated CSV files
