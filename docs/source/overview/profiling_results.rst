.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

Profiling results
=================

Call ``result.to_dataframe()`` on the
:class:`~nsight.collection.core.ProfileResults` returned by
:func:`nsight.analyze.kernel` to inspect the processed results. Each row
summarizes one annotation, metric, and configuration across repeated runs.
Function parameters appear as configuration columns, such as ``n`` in the
:doc:`quickstart`.

Measurement and statistics columns
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Meaning
   * - ``Annotation``, ``Metric``
     - Annotated region and collected or derived metric.
   * - ``AvgValue``
     - Arithmetic mean of the valid metric samples, or the ratio to a baseline
       when normalization is requested (see below).
   * - ``StdDev``
     - Sample standard deviation, using a denominator of ``NumRuns - 1``.
   * - ``MinValue``, ``MaxValue``
     - Minimum and maximum valid sample values.
   * - ``NumRuns``
     - Number of non-missing metric samples used in the aggregation. This can
       be smaller than the requested ``runs`` when samples are missing.
   * - ``CI95_Lower``, ``CI95_Upper``
     - Approximate 95% confidence bounds, computed as
       ``AvgValue +/- 1.96 * StdDev / sqrt(NumRuns)`` before normalization.
       These use a normal approximation, including for small sample counts.
   * - ``RelativeStdDevPct``
     - ``100 * StdDev / abs(AvgValue)``, computed before normalization. It is
       ``NaN`` when the mean is zero or the standard deviation is undefined.
   * - ``StableMeasurement``
     - ``True`` exactly when ``RelativeStdDevPct < 2.0``. This is a variability
       heuristic, not a guarantee of benchmark accuracy. Undefined relative
       variability yields ``pd.NA`` (displayed as ``<NA>``), the missing value
       in a pandas nullable Boolean column.
   * - ``Kernel``, ``GPU``, ``Host``
     - Kernel name and execution environment.
   * - ``ComputeClock``, ``MemoryClock``, ``Unit``
     - Clock metadata and the original metric unit. Availability depends on
       the profiling backend.

One run and missing samples
---------------------------

The default is ``runs=1``. One valid sample is enough to report a value, but
not enough to estimate sample standard deviation. ``StdDev``,
``RelativeStdDevPct``, and both confidence bounds are therefore ``NaN``.
``StableMeasurement`` is ``pd.NA`` in this case, distinguishing unknown
stability from an observed relative standard deviation at or above 2%.

Use ``@nsight.analyze.kernel(runs=10)`` to collect repeated measurements.
At least two valid samples are needed for sample standard deviation. Even
with repeated runs, a zero mean leaves ``RelativeStdDevPct`` undefined.
Missing samples are excluded from the count and standard aggregations; a group
with no valid samples has ``NumRuns=0`` and undefined statistics.

Normalization and geometric mean
--------------------------------

Pass ``normalize_against="baseline"`` to :func:`nsight.analyze.kernel` to
normalize against an annotation named ``baseline`` for the same metric and
configuration.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Meaning
   * - ``Normalized``
     - ``False`` when normalization is disabled; otherwise the name of the
       baseline annotation. It is not always a Boolean.
   * - ``NormalizationValue``
     - Baseline's unnormalized mean. Present only when normalization is enabled.
   * - ``Geomean``
     - Geometric mean of ``AvgValue`` across configurations for the same
       annotation and metric, repeated on each row of that group. It is
       computed after normalization, if enabled. With one positive value,
       it equals that value. Missing values are excluded; negative values are
       outside the real-valued geometric mean's domain.

For example, an unnormalized mean of 20 ns and a baseline mean of 10 ns
produce ``AvgValue=2.0``, ``NormalizationValue=10.0``, and
``Normalized="baseline"``. The ratio is dimensionless. The current
implementation normalizes only ``AvgValue``: ``StdDev``, ``MinValue``,
``MaxValue``, confidence bounds, and ``Unit`` retain their original scale or
metadata. ``RelativeStdDevPct`` and ``StableMeasurement`` describe the
unnormalized samples. Do not interpret the original confidence bounds as
bounds on the normalized ratio.

Custom information collectors can add further columns. See
:doc:`/analyze` for the experimental collector API.
