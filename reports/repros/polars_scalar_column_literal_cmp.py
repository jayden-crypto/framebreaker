import polars as pl

df = pl.DataFrame({"a": [1, 2, 3]}).with_columns(s=pl.lit(1))

print(df.select(pl.lit(0) == pl.col("s")).height)   # 1  (expected 3)
print(df.select(pl.col("s") == pl.lit(0)).height)   # 3
print(df.lazy().select(pl.lit(0) == pl.col("s")).collect(engine="streaming").height)  # 3

# Next to another column it becomes an error instead of a silently short result:
df.with_columns(x=pl.lit(0) == pl.col("s"))
# InvalidOperationError: Series x, length 1 doesn't match the DataFrame height of 3
