<!-- file at https://github.com/pola-rs/polars/issues/new -->
# `lit(x) <cmp> col` returns a length-1 result when the column was created from a literal (regression in 1.14.0)

## Description

If a DataFrame column was created from a literal (e.g. `with_columns(s=pl.lit(1))`), comparing it with a literal **on the left-hand side** (`pl.lit(0) == pl.col('s')`) returns a single row instead of one row per row of the frame. In `select` the result is silently truncated; next to other columns it raises `InvalidOperationError`. Swapping the operands, arithmetic (`pl.lit(0) + pl.col('s')`) and the streaming engine are all correct.

## Reproducible example

```python
import polars as pl

df = pl.DataFrame({"a": [1, 2, 3]}).with_columns(s=pl.lit(1))

print(df.select(pl.lit(0) == pl.col("s")).height)   # 1  (expected 3)
print(df.select(pl.col("s") == pl.lit(0)).height)   # 3
print(df.lazy().select(pl.lit(0) == pl.col("s")).collect(engine="streaming").height)  # 3

# Next to another column it becomes an error instead of a silently short result:
df.with_columns(x=pl.lit(0) == pl.col("s"))
# InvalidOperationError: Series x, length 1 doesn't match the DataFrame height of 3
```

## Output

```
1
3
3
polars.exceptions.InvalidOperationError: Series x, length 1 doesn't match the DataFrame height of 3

If you want expression: [(0) == (col("s"))] to be broadcasted, ensure it is a scalar (for instance by adding '.first()').
```

## Expected behavior

`pl.lit(0) == pl.col('s')` has the frame's height (3), like `pl.col('s') == pl.lit(0)`.

## Notes

- Affects `==`, `!=`, `<`, `<=`, `>`, `>=` with any literal dtype (int, bool, str), eager and lazy, with or without optimizations; the streaming engine is correct.
- Bisected: correct in 1.13.1, broken from 1.14.0 through 1.44.2.
- Possibly related to #29644 (`clip` with a length-1 input), but that issue states other scalar-receiver ops broadcast correctly; comparisons against a scalar-backed column don't.

## Versions

polars 1.44.2, pandas 3.0.6, pyarrow 25.0.1, duckdb 1.5.6, narwhals 2.26.0, Python 3.13.15

---
Found with differential fuzzing (Polars/pandas/DuckDB/PyArrow via Narwhals) and minimised automatically.
