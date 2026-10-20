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
