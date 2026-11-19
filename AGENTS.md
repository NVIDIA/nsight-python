# Agent instructions

Nsight Python (`nsight-python` on PyPI, imported as `nsight`) profiles GPU
kernels from Python. You decorate a benchmark function with
`@nsight.analyze.kernel`, mark kernel launches with `nsight.annotate`, and get
back a `ProfileResults` whose `.to_dataframe()` holds aggregated metrics.
Collection is done by NVIDIA Nsight Compute (`ncu`, default) or, for kernel
time only, by the experimental CUPTI backend.

This file has two parts:

- **Part 1: Using Nsight Python.** How the profiler behaves and how to write
  and read benchmarks. Read this whenever you write or run profiling code,
  including tests and examples in this repository.
- **Part 2: Working on this repository.** Layout, commands and rules for
  changing the library.

## Sources of truth

This file records rules and pitfalls. Facts that change between releases live
elsewhere. Check them there rather than relying on memory or on this file. If
this file disagrees with them, they win, and this file should be fixed.

| Topic | Where to look |
| --- | --- |
| Required `ncu`, `ncu-report` and CUPTI versions | `docs/source/installation/runtime_requirements.rst`; enforced by `MIN_NCU_VERSION` in `nsight/collection/ncu.py` |
| Decorator and annotation parameters | Docstrings of `kernel` in `nsight/analyze.py` and `annotate` in `nsight/annotation.py` |
| Result columns and their meaning | `docs/source/overview/profiling_results.rst`, `ProfileResults.to_dataframe` |
| Output files and CSV columns | `kernel` docstring in `nsight/analyze.py` |
| Backends, CUPTI limitations | `docs/source/tools.rst` |
| Known issues | `docs/source/release_notes/known_issues.rst` |
| Metric names | `ncu --query-metrics --chip <chip>` (for example `ga102`, `gh100`) |
| Dev commands, lint and test setup | `pyproject.toml` (hatch envs), `conftest.py` |
| CI jobs and GPU matrix | `.github/workflows/ci.yaml` |
| Commit and PR rules | `Contributing.md` |

Do not invent parameters, result columns, metric names, or units.

# Part 1: Using Nsight Python

## Before running anything

- Check `nvidia-smi` and `ncu --version` against the runtime requirements.
  Many machines have an older `ncu` from a CUDA toolkit first on `PATH`. List
  every `ncu` on `PATH` with `which -a ncu` (Linux) or `where.exe ncu`
  (Windows).
- An older or missing `ncu` does not fail at `import nsight`. The error is
  stored and raised on the first call to a decorated function.
- Non-admin profiling needs access to GPU performance counters. If they are
  restricted to administrators, `ncu` fails with `ERR_NVGPUCTRPERM`. On Linux
  you can check beforehand: `/proc/driver/nvidia/params` shows
  `RmProfilingAdminOnly: 0` when access is open. On Windows the setting is in
  the NVIDIA Control Panel (Developer > Manage GPU Performance Counters). Do
  not change driver settings or the Control Panel setting, add capabilities,
  or run as root or administrator to get around this. Report it instead.
- On Windows, set `$env:MPLBACKEND = "Agg"` (PowerShell) before using
  `@nsight.analyze.plot` in a non-interactive run. matplotlib defaults to the
  TkAgg GUI backend there, and plotting fails where Tk is unusable, as on
  the CI runners. On Linux, matplotlib falls back to `Agg` by itself when
  there is no display.
- On Windows, a Python process that loaded NCU can abort at exit with code
  `0xC0000409` after all its work has finished. Judge success from the
  program's own output (for example pytest's summary line), not only from
  the exit code. CI works around this in `.github/workflows/ci.yaml`.
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
   `RuntimeError: More than one ... kernel is launched`. Options are
   `combine_kernel_metrics=lambda a, b: a + b`, `ignore_kernel_list=[...]`, or
   `replay_mode="range"`. Every annotation must launch the **same number of
   kernels in every run and config**. The check is only
   `total % (configs * runs) == 0`, so varying counts can pass and give wrong
   numbers.
5. **NCU replays each kernel** with caches flushed (`cache_control="all"`,
   the default; `"none"` keeps caches warm) and measures it in isolation.
   `gpu__time_duration.sum` is per-kernel GPU time in ns. It does not
   include launch overhead or host time, and it is not end-to-end
   wall-clock.
6. Output goes to a fresh `nspy_*` directory in the system temp directory
   unless `output_prefix` is set.

## Warm up first-call work

Code that does extra work on its first call breaks the
one-kernel-per-annotation rule and makes runs non-uniform. Examples are Triton
`@triton.autotune` (which benchmarks every config, launching dozens of
kernels), `torch.compile`, and library heuristics that probe on first use.

- Before the decorated call, run the code once for **every shape** in
  `configs`, outside profiling. The kernel-count errors suggest this too.
  Follow that suggestion rather than reaching for `combine_kernel_metrics`,
  range replay or `ignore_kernel_list`, which would hide the warm-up kernels
  in the result instead of removing them.
- Triton JIT compilation launches no kernels and is safe inside an
  annotation, but warming up outside keeps runs uniform.

## Kernel names

- The `Kernel` column holds the launched kernel's name, for example the
  `@triton.jit` function name. Variants of one kernel (different
  `num_warps`, block sizes or template arguments) can share a name, so encode
  the variant in the annotation name (`f"triton_bm{bm}_w{w}"`).
- Names are **demangled** under NCU (`vectorized_elementwise_kernel`) but
  **mangled** under CUPTI (`_ZN2at6native29vectorized_elementwise_kernel...`).
  `ignore_kernel_list` matches exact names, so write it for the backend in use.

## Failures inside annotations

- By default, an exception inside an annotation stops the whole profiling
  call, so one failing candidate aborts a sweep.
- With `nsight.annotate(name, ignore_failures=True)`, the failed run gets
  `NaN`, so the row can show `AvgValue=NaN`, `NumRuns=0` and
  `Kernel="dummy_kernel_failure"`. The exception itself is reported only as a
  `RuntimeWarning` ("Ignored failure in annotation ..."), so capture warnings
  if you need to know why a candidate failed.

## CUPTI backend (experimental)

Opt in by setting `NSPY_NCU_INIT_AT_IMPORT=0` before `import nsight` (in the
shell, or with `os.environ` at the top of the script), then call
`nsight.activate(nsight.Tool.CUPTI)`. It has lower overhead than NCU for
timing-only loops. Check `docs/source/tools.rst` for supported platforms
(currently not Windows), metrics and limitations. Rules that are not written
down there:

- **Activate CUPTI before importing torch.** If `torch` is imported before
  `nsight.activate(nsight.Tool.CUPTI)` *and* a CUDA context is created before
  the first decorated call, no kernel launch records arrive. Any torch op,
  tensor allocation, Triton launch or `torch.cuda.init()` creates a context.
  Every profiled call in that process then raises `RuntimeError: No kernels
  were profiled`, even though the annotated code did launch kernels. Warm-up
  and correctness checks can run anywhere once CUPTI is active. Importing
  Triton or `cuda.core` first is fine. (Observed with cupti-python 13.4 and
  PyTorch 2.12.)
- CUPTI ignores `cache_control`, so caches are not flushed between runs and
  short kernels vary much more than under NCU. Check `RelativeStdDevPct`
  before trusting a difference.

## Reading results

`ProfileResults.to_dataframe()` returns one row per
(annotation, config, metric). Inspect `df.columns` and the results docs
rather than assuming columns. Interpretation rules:

- **`NaN` means failed or unavailable, not zero.** `NumRuns` counts valid
  samples only.
- **With `normalize_against`, only `AvgValue` is normalized.** `Unit` still
  shows the original unit, for example `ns`, and the spread columns stay in
  raw units. For comparisons, prefer computing ratios yourself from
  un-normalized data. If you do use it on a time metric, speedup is
  `1 / AvgValue`.
- `derive_metric(m1, ..., mk, *config_args)` receives metrics in the order
  given in `metrics=`, followed by the function's parameters. Return
  `(value, unit)` or `{name: (value, unit)}`. Bare scalars get `Unit=NaN` and
  a warning.
- With `runs=1` (the default), there is no spread estimate. Use `runs >= 3`
  before claiming a difference, and compare it against `RelativeStdDevPct`.
- Throughput metrics (`*.pct_of_peak_sustained_elapsed`, unit `%`) tell you
  whether a kernel is memory- or compute-bound. Collect them alongside time
  rather than inferring a bottleneck from time alone.
- Do not read plot PNGs to make decisions. Everything `@nsight.analyze.plot`
  shows comes from the same DataFrame.

## Writing benchmarks

Use this shape. Warm up and validate first, outside profiling, then profile
with a small, bounded configuration set:

```python
import torch  # importing torch does not initialize CUDA

import nsight  # must be imported before the first CUDA call

SHAPES = [(4096, 1024), (4096, 4096)]


def my_softmax(x: torch.Tensor) -> torch.Tensor: ...  # the candidate to evaluate


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


def main() -> None:
    # Warm up and validate every shape outside profiling. assert_close picks
    # tolerances from the dtype; pass atol/rtol if the candidate needs looser.
    for rows, cols in SHAPES:
        x = torch.randn(rows, cols, device="cuda")
        torch.testing.assert_close(my_softmax(x), torch.softmax(x, dim=1))

    df = bench().to_dataframe()  # configs supply the arguments
    columns = ["Annotation", "rows", "cols", "Metric", "AvgValue", "RelativeStdDevPct"]
    print(df[columns].to_string(index=False))


if __name__ == "__main__":
    main()
```

- A candidate written as several PyTorch ops launches several kernels and
  raises the more-than-one-kernel error. To time it as a whole, pass
  `combine_kernel_metrics=lambda a, b: a + b`. Summing is right for times and
  counts, but not for percentages or rates such as
  `*.pct_of_peak_sustained_elapsed`, so profile those per kernel or in a
  separate run.
- Validate every candidate with a dtype-appropriate tolerance **before**
  accepting its timing. Report incorrect candidates separately from profiling
  failures. Faster but wrong is not an improvement.
- Each extra NCU metric can add replay passes. A sweep's cost is roughly
  configs × runs × annotations × passes. Start with one config and
  `runs=1` as a smoke test.
- Check metric names before a long run. An invalid name only raises
  `ProfilerException: Invalid value ...` **after** the whole workload has run.
- If no annotation launches a kernel, NCU writes no report and you get
  `ProfilerException: No NVIDIA Nsight Compute report found`. Check that
  every annotation really launches a kernel (views, no-ops and cached results
  do not).

## GPU execution rules

- Reading and editing code does not need a GPU.
- Assume the GPU is shared. Do not kill other processes, lock clocks with
  `nvidia-smi`, change power limits, or start background profiling jobs. Use
  `clock_control="base"` (via ncu) only if asked.
- Run with a time limit: your tool's command timeout, or
  `timeout 600 python ...` on Linux. Windows has no direct equivalent
  (`timeout.exe` only waits), so rely on the tool's timeout there.
  Smoke-test with one config before a sweep.
- If the GPU or `ncu` is unavailable, run lint and any CPU-only checks, then
  state exactly what was not verified. Never report a GPU test as passing
  without having run it.

# Part 2: Working on this repository

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
| `conftest.py` | Adds `--tool-init`, which selects how the profiling tool is set up. |
| `docs/source/` | Sphinx docs. Release notes in `docs/source/release_notes/`. |

## Commands

The hatch environments are defined in `pyproject.toml`. CI runs the same ones.

Tests and examples also need pytest and a CUDA build of PyTorch, which are
not package dependencies (CI installs PyTorch separately; the hatch test
environments use system packages). Some examples need Triton or cuTile and
are skipped without them.

```bash
pip install -e ".[cu13]"           # or [cu12]; add ,cupti for the CUPTI backend
pip install pytest pytest-isolated torch           # pick the torch build matching your CUDA, see pytorch.org
hatch run lint:check               # black --check, isort --check, mypy --strict (whole repo, incl. examples/tests)
hatch run lint:format              # apply black + isort
hatch run docs:build               # sphinx-build -W (warnings are errors)
pytest tests -v                    # unit/integration tests (GPU + ncu)
pytest examples -v                 # runs every example's main()
pytest tests/test_profiler.py::test_name -v   # one test
pytest -v --tool-init=cupti        # same suites on the CUPTI backend (not on Windows)
hatch run test_cu13:run            # what CI runs
```

- `mypy --strict` covers `examples/` and `tests/` too. Annotate every
  function. Use the existing `# type: ignore[untyped-decorator]` on
  `@triton.jit` and `@pytest.mark.*` decorators.
- `scripts/git-hooks/pre-commit` runs `hatch run lint:check`. Enable it with
  `git config core.hooksPath scripts/git-hooks`, unless the user has their own
  hooks set up.
- Follow `Contributing.md` for commit sign-off and title format.

## Adding an example

- Use the next free number, `NN_snake_case.py`. Include the SPDX header,
  a module docstring with the concepts it introduces, and a typed `main()`
  under `if __name__ == "__main__":`.
- Add a `test_NN_*` that imports it and calls `main()` in
  `examples/test_examples.py`. Use `pytest.importorskip` for optional
  frameworks (Triton, cuTile), and the `_skip_cupti` marker if it needs NCU
  metrics.
- Add it to `examples/README.md`, including any extra install step.
- Keep default workloads small. `pytest examples` runs every example in CI,
  on several GPUs and on both Linux and Windows.

## Changing library code

- Public behavior changes need a docstring update in `nsight/analyze.py` (the
  docs are autodoc-generated), relevant `docs/source/*.rst` pages, and a
  release-note entry under `docs/source/release_notes/topics/`.
- New result columns or output formats must be documented in the `kernel`
  docstring (CSV section), `ProfileResults.to_dataframe` and
  `docs/source/overview/profiling_results.rst`.
- New limitations go in `docs/source/release_notes/known_issues.rst` or the
  relevant docs page, not only in this file.
- Add tests in `tests/`. Use `nsight.get_active_tool() == nsight.Tool.CUPTI`
  with a `skipif` for NCU-only behavior, as `tests/test_profiler.py` does.
  Do not weaken assertions to make a test pass.
- Don't add runtime dependencies without justification. The core deps are
  listed in `pyproject.toml`.

# Completion report

Summarize what changed and why, which commands you ran and their outcome,
whether GPU profiling actually ran (and on which GPU and `ncu` version), and
anything not verified.
