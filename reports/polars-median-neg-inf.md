<!-- file at https://github.com/pola-rs/polars/issues/new -->
# `median` / `quantile(interpolation='linear')` returns NaN when the lower middle value is -inf

## Description

With an even number of values whose lower middle value is `-inf`, `median()` (and `quantile(0.5)` with linear interpolation) returns `NaN`. The symmetric case with `+inf` as the upper middle value correctly returns `inf`. NumPy, DuckDB and pandas return `-inf`. Same result in `select`, `group_by().agg()` and the streaming engine; `quantile(0.5, interpolation='midpoint')` correctly returns `-inf`.

## Reproducible example

```python
import duckdb
import numpy as np
import polars as pl

for vals in ([0.0, float("-inf")], [1.0, float("inf")], [1.0, 2.0, float("-inf"), 5.0]):
    s = pl.Series(vals)
    ddb = duckdb.execute("SELECT median(x) FROM (SELECT unnest(?) AS x)", [vals]).fetchone()[0]
    print(f"{vals}: polars median={s.median()}  quantile(0.5, 'linear')={s.quantile(0.5, 'linear')}  "
          f"numpy={np.median(vals)}  duckdb={ddb}")
# [0.0, -inf]: polars median=nan  quantile(0.5, 'linear')=nan  numpy=-inf  duckdb=-inf   <- wrong
# [1.0, inf]:  polars median=inf  quantile(0.5, 'linear')=inf  numpy=inf   duckdb=inf    <- fine

print(pl.DataFrame({"k": [1, 1], "x": [0.0, float("-inf")]}).group_by("k").agg(pl.col("x").median()))
```

## Output

```
[0.0, -inf]: polars median=nan  quantile(0.5, 'linear')=nan  numpy=-inf  duckdb=-inf
[1.0, inf]: polars median=inf  quantile(0.5, 'linear')=inf  numpy=inf  duckdb=inf
[1.0, 2.0, -inf, 5.0]: polars median=1.5  quantile(0.5, 'linear')=1.5  numpy=1.5  duckdb=1.5
shape: (1, 2)
┌─────┬─────┐
│ k   ┆ x   │
│ --- ┆ --- │
│ i64 ┆ f64 │
╞═════╪═════╡
│ 1   ┆ NaN │
└─────┴─────┘
```

## Expected behavior

`-inf` (the mean of the two middle values, `(-inf + 0.0) / 2`).

## Notes

- Likely cause: linear interpolation computed as `lo + (hi - lo) * t`. With `lo = -inf` and finite `hi`, `hi - lo = inf` and `-inf + inf = NaN`. Computing `lo * (1 - t) + hi * t`, or returning `lo` when `lo == hi` or `lo` is infinite, avoids it.
- Found because the differential fuzzer saw Polars alone disagree with DuckDB, PyArrow and pandas on a grouped median.

## Versions

polars 1.44.2, pandas 3.0.6, pyarrow 25.0.1, duckdb 1.5.6, narwhals 2.26.0, Python 3.13.15

---
Found with differential fuzzing (Polars/pandas/DuckDB/PyArrow via Narwhals) and minimised automatically.
