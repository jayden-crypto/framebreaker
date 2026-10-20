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
