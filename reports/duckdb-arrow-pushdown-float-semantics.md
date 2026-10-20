<!-- file at https://github.com/duckdb/duckdb-python/issues/new -->
# Wrong results: filters pushed into Arrow scans use Arrow float semantics (NaN, -0.0) instead of DuckDB's

## Description

When a filter is pushed down into an Arrow scan (a registered pyarrow Table, or a Polars DataFrame via replacement scan), it is evaluated with Arrow/IEEE semantics instead of DuckDB's. Rows then silently disappear:

- `WHERE x > 0` drops NaN rows, although in DuckDB NaN is greater than every number, and `SELECT x > 0` on the same table returns `true` for them.
- `WHERE x IN (0.0, 7.0)` drops `-0.0` rows, although `-0.0 = 0.0` in DuckDB.
- The dynamic filters created by join filter pushdown have the same problem, so plain equi-joins on float keys lose matches between `-0.0` and `0.0`.

The same queries on native DuckDB tables (and on pandas DataFrames) are correct. A query's result therefore depends on whether the optimizer decides to push a filter down.

## Reproducible example

```python
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
```

## Output

```
SELECT x FROM t WHERE x > 0 ORDER BY x
  native     [(1.0,), (nan,)]
  arrow      [(1.0,)]
  polars_df  [(1.0,)]
SELECT x FROM t WHERE x IN (0.0, 7.0) ORDER BY x
  native     [(0.0,)]
  arrow      []
  polars_df  []
[(0.0, False, True), (1.0, True, False), (nan, True, False)]
[]
[(1.0,), (nan,)]
[(-0.0, None, 0.0, '')]
```

## Expected behavior

The same rows as on a native table: `[(1.0,), (nan,)]` and the `-0.0` row respectively.

## Notes

- Disabling `filter_pushdown` / `join_filter_pushdown` restores correct results.
- Likely cause: pushed-down filters are translated to `pyarrow.compute` expressions. Arrow comparisons follow IEEE (NaN compares false) and `is_in` hashes -0.0 and 0.0 differently (apache/arrow#26214): `pc.is_in(pa.array([-0.0]), value_set=pa.array([0.0]))` is `[False]`.
- Possible fixes: don't push float comparison/IN filters into Arrow scans, or keep them as a residual DuckDB filter (push down a superset), or normalise NaN/-0.0 when translating.
- Related but different symptoms in the same code path: #615 (NULL rows pass a pushed-down `<= 'nan'` filter) and duckdb/duckdb#16942 (filtering *for* NaN in a Polars scan). This report covers NaN *column values* against ordinary constants, and -0.0 in IN-lists and join filters.

## Versions

polars 1.44.2, pandas 3.0.6, pyarrow 25.0.1, duckdb 1.5.6, narwhals 2.26.0, Python 3.13.15

---
Found with differential fuzzing (Polars/pandas/DuckDB/PyArrow via Narwhals) and minimised automatically.
