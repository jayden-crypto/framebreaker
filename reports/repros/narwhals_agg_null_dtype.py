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
