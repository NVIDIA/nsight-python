# Agent instructions

Nsight Python (`nsight-python` on PyPI, imported as `nsight`) profiles GPU
kernels from Python. You decorate a benchmark function with
`@nsight.analyze.kernel`, mark kernel launches with `nsight.annotate`, and get
back a `ProfileResults` whose `.to_dataframe()` holds aggregated metrics.
Collection is done by NVIDIA Nsight Compute (`ncu`, default) or, for kernel
time only, by the experimental CUPTI backend.

## Repository map

| Path | What it is |
| --- | --- |
| `nsight/analyze.py` | Public decorators: `kernel`, `plot`, `ignore_failures`. Docstrings here are the API reference. |
| `nsight/annotation.py` | `nsight.annotate` (context manager and decorator). |
| `nsight/collection/core.py` | Config handling, the run loop (`run_profile_session`), `ProfileResults`. |
| `nsight/collection/ncu.py` | NCU injection loading, `ncu --mode attach` launch, version checks. |
| `nsight/collection/cupti.py` | CUPTI activity-based timing backend. |
| `nsight/extraction.py` | Report to raw DataFrame: kernel-per-annotation checks, `derive_metric`. |
| `nsight/transformation.py` | Aggregation into `AvgValue`/`StdDev`/CI columns, `normalize_against`, `Geomean`. |
| `nsight/visualization.py` | Matplotlib plotting used by `@nsight.analyze.plot`. |
| `nsight/info_collector.py`, `nsight/experimental.py` | Custom info collectors (`@nsight.experimental.collect`). |
| `nsight/thermovision.py` | GPU thermal-headroom throttling guard. |
| `nsight/tools_manager.py` | `nsight.activate` / `deactivate` / `get_active_tool`, `nsight.Tool`. |
| `examples/` | Numbered runnable examples, indexed in `examples/README.md`, run by `examples/test_examples.py`. |
| `tests/` | Pytest suite. Most tests need a GPU and `ncu`. |
| `conftest.py` | Adds `--tool-init {ncu-at-import,ncu-lazy,cupti}`. |
| `docs/source/` | Sphinx docs. Release notes in `docs/source/release_notes/`. |

Read the docstring of `nsight.analyze.kernel` before using or changing the
API. Do not invent parameters, result columns, metric names, or units; check
them against the code, or against `ncu --query-metrics` for metric names.

## Commands

All of these are defined in `pyproject.toml` and `.github/workflows/ci.yaml`.

```bash
pip install -e ".[cu13]"           # or [cu12]; add ,cupti for the CUPTI backend
hatch run lint:check               # black --check, isort --check, mypy --strict (whole repo, incl. examples/tests)
hatch run lint:format              # apply black + isort
hatch run docs:build               # sphinx-build -W (warnings are errors)
pytest tests -v                    # unit/integration tests (GPU + ncu)
pytest examples -v                 # runs every example's main()
pytest -v --tool-init=cupti        # same suites on the CUPTI backend
hatch run test_cu13:run            # what CI runs (cu13 + cupti extras)
```

- `mypy --strict` covers `examples/` and `tests/` too. Annotate every
  function. Use the existing `# type: ignore[untyped-decorator]` on
  `@triton.jit` and `@pytest.mark.*` decorators.
- `scripts/git-hooks/pre-commit` runs `hatch run lint:check`.
- Commits need DCO sign-off (`git commit -s`). See `Contributing.md` for the
  title format.

## Runtime requirements and environment checks

- NVIDIA Nsight Compute **2026.2.1 or newer** must be the `ncu` on `PATH`
  (`MIN_NCU_VERSION` in `nsight/collection/ncu.py`). `ncu-report` must not be
  older than `ncu`. Many machines have an older `ncu` from a CUDA toolkit
  first on `PATH`. Check `ncu --version` before running anything.
- An older `ncu`, or a missing one, does not fail at `import nsight`. The
  error is stored and raised on the first call to a decorated function.
- Non-admin profiling needs GPU performance counters unrestricted
  (`RmProfilingAdminOnly: 0` in `/proc/driver/nvidia/params`). Do not change
  driver settings, add capabilities, or use `sudo` to get around this. Report
  it instead.
- Headless runs: set `MPLBACKEND=Agg` when using `@nsight.analyze.plot`.
- `UserWarning: Nsight Python Thermovision is not supported on this machine`
  is harmless. Thermal throttling protection is simply off on that GPU.

## How the profiler runs your code

These facts decide how to write benchmark functions correctly:

1. **NCU must attach before CUDA is initialized.** `import nsight` activates
   NCU by default (`NSPY_NCU_INIT_AT_IMPORT=1`), so import `nsight` before
   anything that touches CUDA. Only one tool can be active per process, and
   NCU can never be deactivated.
2. **The decorated function is called `len(configs) * runs` times.** Its
   return value is discarded (a `RuntimeWarning` is emitted if it returns
   something). The call returns `ProfileResults`. Keep anything whose result
   you need, such as correctness checks, in an undecorated function.
3. **Only kernels inside `nsight.annotate(...)` are measured.** Setup such as
   `torch.randn` outside annotations is not profiled. Annotation names must be
   unique per run, and annotations cannot be nested.
4. **One kernel per annotation by default.** More than one raises
   `RuntimeError: More than one (total=N) kernel is launched...`. Options are
   `combine_kernel_metrics=lambda a, b: a + b`, `ignore_kernel_list=[...]`, or
   `replay_mode="range"`. Every annotation must launch the **same number of
   kernels in every run and config**. The check is only
   `total % (configs * runs) == 0`, so varying counts can pass and give wrong
   numbers.
5. **NCU replays each kernel** with caches flushed (`cache_control="all"`)
   and measures it in isolation. `gpu__time_duration.sum` is per-kernel GPU
   time in ns. It does not include launch overhead or host time, and it is
   not end-to-end wall-clock.
6. Output files go to a fresh `$TMPDIR/nspy_*` directory unless
   `output_prefix` is set. `output_csv=True` writes `profiled_data-*.csv`
   (raw, one row per run) and `processed_data-*.csv` (same as
   `to_dataframe()`).

### Triton-specific pitfalls

- **Autotuning inside an annotation breaks profiling.** `@triton.autotune`
  benchmarks every config on first call, launching dozens of kernels. In a
  test with a 4-config autotuner this produced 58 kernels over 2 runs in one
  annotation, which raises `RuntimeError: More than one ... kernel is launched`.
  Before the decorated call, call the kernel once for **every shape** in
  `configs` so the autotune cache is warm. Don't use the error's other
  suggestions (`combine_kernel_metrics`, range replay) for this case.
- JIT compilation on first call launches no kernels and is safe inside an
  annotation, but warming up outside keeps runs uniform.
- Kernel names in results are the `@triton.jit` function name, for example
  `softmax_kernel`. Different `num_warps`/block variants of one kernel share a
  name, so name annotations by variant (`f"triton_bm{bm}_w{w}"`).
- A Triton compile error inside an annotation propagates unchanged. With
  `nsight.annotate(name, ignore_failures=True)`, the row gets `AvgValue=NaN`,
  `NumRuns=0` and `Kernel="dummy_kernel_failure"`. The ignored exception is
  reported only as a `RuntimeWarning` ("Ignored failure in annotation ..."),
  so capture warnings if you need to know why a candidate failed.

### CUPTI backend (experimental, Linux only)

Opt in with `NSPY_NCU_INIT_AT_IMPORT=0` before `import nsight`, then call
`nsight.activate(nsight.Tool.CUPTI)`. It has lower overhead than NCU for
timing-only loops (no replay, no `ncu` subprocess), but:

- It supports only `gpu__time_duration.sum`. There is no replay, no cache
  flush and no clock control.
- **Activate CUPTI before importing torch.** Observed with nsight-python
  1.1.0, cupti-python 13.4 and PyTorch 2.12: if `torch` is imported before
  `nsight.activate(nsight.Tool.CUPTI)` *and* a CUDA context is created before
  the first decorated call, no driver launch records arrive. Any torch op,
  tensor allocation, Triton launch or `torch.cuda.init()` creates a context.
  Every profiled call in that process then raises `RuntimeError: No kernels
  were profiled`, even though the annotated code did launch kernels. Set the
  environment variable and activate CUPTI before `import torch`. Then warm-up
  and correctness checks can run anywhere. Importing Triton or `cuda.core`
  first is fine.
- Kernel names in the `Kernel` column are **mangled** under CUPTI (for
  example `_ZN2at6native29vectorized_elementwise_kernel...`) but demangled
  under NCU (`vectorized_elementwise_kernel`). `ignore_kernel_list` matches
  exact names, so write it for the backend in use.
- Without cache flushing, short kernels can vary a lot between runs: about 32%
  relative stddev was seen for a 10 µs kernel. Check `RelativeStdDevPct`.

## Reading results

`ProfileResults.to_dataframe()` returns one row per
(annotation, config, metric). Inspect `df.columns` rather than assuming them.
They currently include:

- `Annotation`, one column per decorated-function parameter, and `Metric`
  (the raw metric name or `derive_metric` names).
- `AvgValue`, `StdDev`, `MinValue`, `MaxValue`, `NumRuns`, `CI95_Lower`,
  `CI95_Upper`, `RelativeStdDevPct`, and `StableMeasurement` (nullable; True
  when the relative stddev is under 2%).
- `Unit`, `Kernel` (first kernel name for the group), `GPU`, `Host`,
  `ComputeClock`, `MemoryClock`, `Normalized`, `Geomean` (geomean of
  `AvgValue` across configs per annotation and metric). `NormalizationValue`
  is added only with `normalize_against`.

Interpretation rules:

- **`NaN` means failed or unavailable, not zero.** `NumRuns` counts valid
  samples only.
- **With `normalize_against`, only `AvgValue` is normalized.** `Unit` still
  shows the original unit, for example `ns`, and `StdDev`, `Min`/`Max` and
  `CI95_*` stay in raw units. For comparisons, prefer computing ratios
  yourself from un-normalized data. If you do use it, compute speedup as
  `1 / AvgValue`.
- `derive_metric(m1, ..., mk, *config_args)` receives metrics in the order
  given in `metrics=`, followed by the function's parameters. Return
  `(value, unit)` or `{name: (value, unit)}`. Bare scalars get `Unit=NaN` and
  a warning.
- With `runs=1`, `StdDev` and `CI95_*` are `NaN`. Use `runs >= 3` before
  claiming a difference, and compare against `RelativeStdDevPct`.
- Throughput metrics (`*.pct_of_peak_sustained_elapsed`, unit `%`) tell you
  whether a kernel is memory- or compute-bound. Collect them alongside time
  rather than inferring a bottleneck from time alone.
- Do not read plot PNGs to make decisions. Everything `@nsight.analyze.plot`
  shows comes from the same DataFrame.

## Writing benchmarks and examples

Use this shape. Validate first, then profile with a small, bounded
configuration set:

```python
import torch  # importing torch does not initialize CUDA

import nsight  # must be imported before the first CUDA call

SHAPES = [(4096, 1024), (4096, 4096)]


def check(candidate, ref) -> None:  # outside profiling
    torch.testing.assert_close(candidate, ref, atol=..., rtol=...)


@nsight.analyze.kernel(
    configs=SHAPES,
    runs=3,
    metrics=[
        "gpu__time_duration.sum",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    ],
)
def bench(rows: int, cols: int) -> None:
    x = torch.randn(rows, cols, device="cuda")
    with nsight.annotate("baseline"):
        torch.softmax(x, dim=1)
    with nsight.annotate("candidate"):
        my_softmax(x)
```

- Validate every candidate with a dtype-appropriate tolerance **before**
  accepting its timing. Report incorrect candidates separately from profiling
  failures. Faster but wrong is not an improvement.
- Each extra NCU metric can add replay passes. A sweep's cost is roughly
  configs × runs × annotations × passes. Start with one config and
  `runs=1` as a smoke test.
- Check metric names first with `ncu --query-metrics --chip <chip>` (for
  example `ga102`, `gh100`). An invalid name only raises
  `ProfilerException: Invalid value ...` **after** the whole workload has run.
- If no annotation launches a kernel, NCU writes no report and you get
  `ProfilerException: No NVIDIA Nsight Compute report found`. Check that
  every annotation really launches a kernel (views, no-ops and cached results
  do not).

When adding an example to `examples/`:

- Use the next free number, `NN_snake_case.py`. Include the SPDX header,
  a module docstring with the concepts it introduces, and a typed `main()`
  under `if __name__ == "__main__":`.
- Add a `test_NN_*` that imports it and calls `main()` in
  `examples/test_examples.py`. Use `pytest.importorskip` for optional
  frameworks (Triton, cuTile), and the `_skip_cupti` marker if it needs NCU
  metrics.
- Add it to `examples/README.md`, including any extra install step.
- Keep default workloads small. `pytest examples` runs all of them in CI on
  H100 and L4.

## Changing library code

- Public behavior changes need a docstring update in `nsight/analyze.py` (the
  docs are autodoc-generated), relevant `docs/source/*.rst` pages, and a
  release-note entry under `docs/source/release_notes/topics/`.
- Add tests in `tests/`. Use `nsight.get_active_tool() == nsight.Tool.CUPTI`
  with a `skipif` for NCU-only behavior, as `tests/test_profiler.py` does.
  Do not weaken assertions to make a test pass.
- New result columns or output formats must be documented in both the
  `kernel` docstring (CSV section) and `ProfileResults.to_dataframe`.
- Don't add runtime dependencies without justification. The core deps are
  listed in `pyproject.toml`.

## GPU execution rules

- Reading and editing code does not need a GPU. Before running profiling,
  check `nvidia-smi`, `ncu --version` (2026.2.1 or newer) and counter
  permissions.
- Assume the GPU is shared. Do not kill other processes, lock clocks with
  `nvidia-smi`, change power limits, or start background profiling jobs. Use
  `clock_control="base"` (via ncu) only if asked.
- Run with a timeout (`timeout 600 python ...`). Smoke-test with one config
  before a sweep.
- If the GPU or `ncu` is unavailable, run lint and any CPU-only checks, then
  state exactly what was not verified. Never report a GPU test as passing
  without having run it.

## Completion report

Summarize what changed and why, which commands you ran and their outcome,
whether GPU profiling actually ran (and on which GPU and `ncu` version), and
anything not verified.
