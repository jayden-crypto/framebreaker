<!-- file at https://github.com/narwhals-dev/narwhals/issues/new -->
# [Bug]: aggregations whose result is null come back null-typed on PyArrow

## Description

On PyArrow, a scalar aggregation that evaluates to null (e.g. `std` of one value, or `min` of an empty column) produces a column of Arrow `null` type (`Unknown` in Narwhals) instead of keeping the aggregation's dtype. Downstream operations then fail: `fill_null(0.0)` raises `ArrowInvalid: Invalid null value`, and a later `group_by(...).agg(std)` raises `ArrowNotImplementedError` for input type null. `pc.min` itself returns a typed `DoubleScalar`, so the type is lost inside Narwhals.

## Reproducible example

```python
import narwhals as nw
import pandas as pd
import polars as pl
import pyarrow as pa

for name, native in [("polars", pl.DataFrame({"x": [0.0]})), ("pyarrow", pa.table({"x": [0.0]})),
                     ("pandas", pd.DataFrame({"x": [0.0]}))]:
    df = nw.from_native(native, eager_only=True)
    out = df.select(nw.col("x").std())          # std of one value is null
    print(f"{name:8} {out.schema}")
# polars   Schema([('x', Float64)])
# pyarrow  Schema([('x', Unknown)])  <- null type, so e.g. a later .fill_null(0.0) or group_by(...).std() fails
# pandas   Schema([('x', Float64)])

empty = nw.from_native(pa.table({"x": pa.array([], type=pa.float64())}), eager_only=True)
print(empty.select(nw.col("x").min()).schema)   # pyarrow: Unknown (null type)  expected Float64
try:
    empty.select(nw.col("x").min()).with_columns(nw.col("x").fill_null(0.0))
except Exception as e:
    print(type(e).__name__, e)                  # ArrowInvalid: Invalid null value
```

## Output

```
polars   Schema({'x': Float64})
pyarrow  Schema({'x': Unknown})
pandas   Schema({'x': Float64})
Schema({'x': Unknown})
ArrowInvalid Invalid null value
```

## Expected behavior

`Float64`, as with Polars (and as with pandas for this case).

## Versions

polars 1.44.2, pandas 3.0.6, pyarrow 25.0.1, duckdb 1.5.6, narwhals 2.26.0, Python 3.13.15

---
Found with differential fuzzing (Polars/pandas/DuckDB/PyArrow via Narwhals) and minimised automatically.
