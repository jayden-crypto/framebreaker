import duckdb
import polars as pl
import pyarrow as pa

con = duckdb.connect()
con.execute("CREATE TABLE native AS SELECT * FROM (VALUES ('nan'::DOUBLE), (-0.0::DOUBLE), (1.0)) v(x)")
arrow = pa.table({"x": [float("nan"), -0.0, 1.0]})
polars_df = pl.DataFrame({"x": [float("nan"), -0.0, 1.0]})

for sql in ["SELECT x FROM {} WHERE x > 0 ORDER BY x",          # DuckDB: NaN > 0 is true
            "SELECT x FROM {} WHERE x IN (0.0, 7.0) ORDER BY x"]:  # DuckDB: -0.0 = 0.0
    print(sql.format("t"))
    for name in ("native", "arrow", "polars_df"):
        print(f"  {name:10}", con.sql(sql.format(name)).fetchall())
# SELECT x FROM t WHERE x > 0 ORDER BY x
#   native     [(1.0,), (nan,)]
#   arrow      [(1.0,)]            <- NaN row dropped
#   polars_df  [(1.0,)]
# SELECT x FROM t WHERE x IN (0.0, 7.0) ORDER BY x
#   native     [(0.0,)]
#   arrow      []                  <- -0.0 row dropped
#   polars_df  []

# The projection agrees with DuckDB semantics; only the pushed-down filter differs:
print(con.sql("SELECT x, x > 0, x IN (0.0, 7.0) FROM arrow ORDER BY x").fetchall())

# Joins are affected through join filter pushdown:
t0 = pa.table({"c0": [3.0, -0.0], "c1": pa.array([None, None], type=pa.float64())})
t1 = pa.table({"c1": [0.0], "r1": [""]})
print(con.sql("SELECT * FROM t0 CROSS JOIN t1 WHERE t1.c1 = t0.c0").fetchall())  # []  expected 1 row

con.execute("SET disabled_optimizers = 'filter_pushdown,join_filter_pushdown'")
print(con.sql("SELECT x FROM arrow WHERE x > 0 ORDER BY x").fetchall())          # [(1.0,), (nan,)]
print(con.sql("SELECT * FROM t0 CROSS JOIN t1 WHERE t1.c1 = t0.c0").fetchall())  # [(-0.0, None, 0.0, '')]
