import duckdb
import narwhals as nw
import polars as pl

empty = duckdb.sql("SELECT 1.0::DOUBLE AS a WHERE false")
print(nw.from_native(empty).select(nw.col("a").n_unique()).collect()["a"].to_list())  # [None]  expected [0]
print(pl.DataFrame({"a": []}, schema={"a": pl.Float64}).select(pl.col("a").n_unique())["a"].to_list())  # [0]
# Generated SQL: count(DISTINCT a) + max(CASE WHEN a IS NULL THEN 1 ELSE 0 END)  -> max() over no rows is NULL
