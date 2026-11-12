# Nsight Python Examples

This directory contains examples demonstrating how to use Nsight Python for profiling and visualizing CUDA kernel performance.

## Prerequisites

### Required

- **Python 3.10+**
- **CUDA-capable GPU**
- **NVIDIA Nsight Compute** (for profiling)
- **NVIDIA Compute Sanitizer with `--nvtx-include` support** (only for
  `14_compute_sanitizer.py`)

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

#### cuTile Python (Optional)

For the cuTile example (`12_cutile.py`):

Refer [Installing cuTile Python](https://docs.nvidia.com/cuda/cutile-python/quickstart.html#installing-cutile-python).

Refer [cuTile Python Prerequisites section](https://docs.nvidia.com/cuda/cutile-python/quickstart.html#prerequisites).

#### Triton (Optional)

For the Triton example (`07_triton_minimal.py`):

```bash
pip install triton
```

## Quick Start

Start with `00_minimal.py`:

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

- **`10_combine_kernel_metrics.py`** - Combining metrics from multiple kernels
  - Using `combine_kernel_metrics` to aggregate metrics from multiple kernels
  - Summing metrics from consecutive kernel executions

- **`11_output_csv.py`** - Outputting to CSV
  - Using `output_csv` parameter to enable/disable CSV file generation
  - Using `output_prefix` to specify output file location and naming
  - Reading and displaying generated CSV files

- **`12_cutile.py`** - Profiling a cuTile kernel
  - Comparing against PyTorch kernel for different problem sizes

- **`13_custom_info_collectors.py`** - Custom information collectors
  - Using `@nsight.experimental.collect(scope=...)`
  - Collecting information once, per configuration, per run, or per annotation
  - Understanding collection scopes and aggregation behavior
  - Tracking execution order and system state during profiling

- **`14_compute_sanitizer.py`** - Checking annotated regions with Compute Sanitizer
  - Selecting memcheck through `nsight.activate(..., options=...)`
  - Self-injecting one backend and attaching one sanitizer frontend
  - Demonstrating an intentional out-of-bounds CUDA memset

  Run it directly with Python; do not prefix the command with
  `compute-sanitizer` because Nsight Python starts the attach frontend:

  ```bash
  python 14_compute_sanitizer.py
  ```

  The example disables legacy import-time NCU activation before importing
  `nsight`. Applications selecting Compute Sanitizer must do the same and must
  call `activate` before importing a CUDA framework or otherwise initializing
  CUDA.

  The Python script executes only once. The sanitizer frontend remains attached
  until the target exits, so frontend-only findings cannot change the target's
  exit code; use an external supervisor when CI requires the frontend's
  `--error-exitcode` result.

  Mark the regions you want to check with `nsight.annotate(...)`. Sanitizer
  activation instruments the current process, so no decorator is needed.
  Call `nsight.sanitize.log_path()` to get the chosen text log path. The file is
  updated while the process runs.

  Compute Sanitizer writes its diagnostics to a text log. By default, Nsight
  Python selects a unique temporary path such as:

  ```text
  /tmp/compute-sanitizer-output-12345-a1b2c3d4.log
  ```

  Pass an explicit log path at activation when automation needs one. The
  path must not already exist. Choose a tool and pass any tool-specific CLI
  arguments as separate tokens (not one shell command string):

  ```python
  nsight.activate(
      nsight.Tool.COMPUTE_SANITIZER,
      options=nsight.ComputeSanitizerOptions(
          log_path="/tmp/sanitizer.log",
          tool=nsight.ComputeSanitizerTool.RACECHECK,
          tool_args=("--racecheck-report", "all"),
      ),
  )
  ```

  The frontend updates the log as findings arrive. Nsight Python leaves the
  log untouched for the user or external tooling to inspect.
