<!-- file at https://github.com/duckdb/duckdb-python/issues/new -->
# `ConstantExpression(float('nan'))` still produces NULL instead of NaN (bind parameters were fixed in duckdb#4623)

## Description

A NaN passed through the relational API becomes SQL NULL, while `'nan'::DOUBLE` in SQL and `float('inf')` via `ConstantExpression` keep their values. Libraries that build queries through the expression API (e.g. Narwhals' `nw.lit(float('nan'))` on DuckDB, which renders as `CAST(NULL AS DOUBLE)`) therefore turn NaN into NULL, which then behaves differently under `IS NULL`, `count`, and comparisons.

## Reproducible example

```python
import duckdb

rel = duckdb.sql("SELECT 1 AS a")
print(duckdb.sql("SELECT 'nan'::DOUBLE AS x").fetchall())                        # [(nan,)]
print(rel.select(duckdb.ConstantExpression(float("nan")).alias("x")).fetchall())  # [(None,)]  expected [(nan,)]
print(rel.select(duckdb.ConstantExpression(float("inf")).alias("x")).fetchall())  # [(inf,)]
```

## Output

```
[(nan,)]
[(None,)]
[(inf,)]
```

## Expected behavior

`ConstantExpression(float('nan'))` evaluates to NaN (a DOUBLE), like `'nan'::DOUBLE`.

## Notes

- duckdb/duckdb#4623 reported the same NaN-to-NULL conversion for bind parameters; that path now preserves NaN (`con.execute('select ?::DOUBLE', [float('nan')])` returns `nan`), but the relational API's `ConstantExpression` still converts it.
- If the conversion is intentional here (pandas-style missing values), documenting it would help.

## Versions

polars 1.44.2, pandas 3.0.6, pyarrow 25.0.1, duckdb 1.5.6, narwhals 2.26.0, Python 3.13.15

---
Found with differential fuzzing (Polars/pandas/DuckDB/PyArrow via Narwhals) and minimised automatically.
