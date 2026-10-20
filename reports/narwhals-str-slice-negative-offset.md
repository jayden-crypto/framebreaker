<!-- file at https://github.com/narwhals-dev/narwhals/issues/new -->
# [Bug]: `str.slice` with a negative offset gives wrong results on DuckDB, PyArrow and pandas

## Description

Compared with Polars (the reference), `str.slice(offset, length)` with a negative offset is wrong in two different ways:

- **PyArrow / pandas**: when `offset + length >= 0` the stop index is computed as `offset + length` (e.g. 0), giving an empty string instead of slicing to the end.
- **DuckDB**: when `|offset| > len(s)` the generated `substr(c, length(c) + offset, ...)` starts at a non-positive position, giving the wrong characters (`'abc'.slice(-5, 3)` returns `'c'`).

## Reproducible example

```python
import duckdb
import narwhals as nw
import pandas as pd
import polars as pl
import pyarrow as pa

for s, offset, length in [("A", -2, 2), ("ab", -3, None), ("abc", -5, 3)]:
    expected = pl.Series([s]).str.slice(offset, length)[0]
    got = {}
    for name, native in [("duckdb", duckdb.sql(f"SELECT '{s}' AS c")),
                         ("pyarrow", pa.table({"c": [s]})),
                         ("pandas", pd.DataFrame({"c": [s]}))]:
        frame = nw.from_native(native).select(nw.col("c").str.slice(offset, length))
        frame = frame.collect() if isinstance(frame, nw.LazyFrame) else frame
        got[name] = frame["c"].to_list()[0]
    print(f"{s!r}.str.slice({offset}, {length}): polars={expected!r}  {got}")
# 'A'.str.slice(-2, 2): polars='A'    {'duckdb': 'A', 'pyarrow': '', 'pandas': ''}
# 'ab'.str.slice(-3, None): polars='ab' {'duckdb': 'a', 'pyarrow': 'ab', 'pandas': 'ab'}
# 'abc'.str.slice(-5, 3): polars='a'  {'duckdb': 'c', 'pyarrow': 'a', 'pandas': 'a'}
```

## Output

```
'A'.str.slice(-2, 2): polars='A'  {'duckdb': 'A', 'pyarrow': '', 'pandas': ''}
'ab'.str.slice(-3, None): polars='ab'  {'duckdb': 'a', 'pyarrow': 'ab', 'pandas': 'ab'}
'abc'.str.slice(-5, 3): polars='a'  {'duckdb': 'c', 'pyarrow': 'a', 'pandas': 'a'}
```

## Expected behavior

Polars: the start clamps to 0 and the slice covers `[len+offset, len+offset+length)` intersected with the string.

## Versions

polars 1.44.2, pandas 3.0.6, pyarrow 25.0.1, duckdb 1.5.6, narwhals 2.26.0, Python 3.13.15

---
Found with differential fuzzing (Polars/pandas/DuckDB/PyArrow via Narwhals) and minimised automatically.
