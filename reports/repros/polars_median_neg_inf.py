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
