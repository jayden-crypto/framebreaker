<!-- file at https://github.com/narwhals-dev/narwhals/issues/new -->
# [Bug]: string methods on a literal receiver fail on PyArrow and pandas

## Description

`nw.lit('x').str.contains(...)`, `.starts_with(...)` and `.ends_with(...)` raise `TypeError: ... only supports str pattern values` on PyArrow and pandas, although the pattern *is* a str: when the receiver is itself a scalar, the extracted pattern is an array rather than a scalar. On PyArrow, `nw.lit('x').str.len_chars().fill_null(0)` raises `AttributeError: 'ChunkedArray' object has no attribute 'as_py'`.

## Reproducible example

```python
import narwhals as nw
import pandas as pd
import polars as pl
import pyarrow as pa

for name, native in [("polars", pl.DataFrame({"a": [1]})), ("pyarrow", pa.table({"a": [1]})),
                     ("pandas", pd.DataFrame({"a": [1]}))]:
    df = nw.from_native(native, eager_only=True)
    for label, expr in [("contains", nw.lit("x").str.contains("x", literal=True)),
                        ("starts_with", nw.lit("x").str.starts_with("x")),
                        ("len_chars().fill_null", nw.lit("x").str.len_chars().fill_null(0))]:
        try:
            out = df.with_columns(r=expr)["r"].to_list()
        except Exception as e:
            out = f"{type(e).__name__}: {e}"
        print(f"{name:8} {label:24} {out}")
# polars:  [True] / [True] / [1]
# pyarrow: TypeError: `.str.contains` only supports str pattern values ... ; AttributeError: 'ChunkedArray' object has no attribute 'as_py'
# pandas:  TypeError: `.str.contains` only supports str pattern values ...
```

## Output

```
polars   contains                 [True]
polars   starts_with              [True]
polars   len_chars().fill_null    [1]
pyarrow  contains                 TypeError: `.str.contains` only supports str pattern values for pyarrow backend
pyarrow  starts_with              TypeError: `.str.starts_with` only supports str prefix values for pyarrow backend
pyarrow  len_chars().fill_null    AttributeError: 'pyarrow.lib.ChunkedArray' object has no attribute 'as_py'
pandas   contains                 TypeError: `.str.contains` only supports str pattern values for pandas backend
pandas   starts_with              TypeError: `.str.starts_with` only supports prefix values for pandas backend
pandas   len_chars().fill_null    [1]
```

## Expected behavior

Polars: the expression broadcasts like any other scalar expression.

## Notes

- Same family as #3964 (scalar-like receiver for `clip`), but a different code path.

## Versions

polars 1.44.2, pandas 3.0.6, pyarrow 25.0.1, duckdb 1.5.6, narwhals 2.26.0, Python 3.13.15

---
Found with differential fuzzing (Polars/pandas/DuckDB/PyArrow via Narwhals) and minimised automatically.
