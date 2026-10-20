<!-- file at https://github.com/narwhals-dev/narwhals/issues/new -->
# [Bug]: `n_unique()` returns NULL instead of 0 on an empty DuckDB relation

## Description

The DuckDB backend renders `n_unique` as `count(DISTINCT x) + max(CASE WHEN x IS NULL THEN 1 ELSE 0 END)`. Over zero rows `max()` is NULL, so the whole expression is NULL instead of 0.

## Reproducible example

```python
import duckdb
import narwhals as nw
import polars as pl

empty = duckdb.sql("SELECT 1.0::DOUBLE AS a WHERE false")
print(nw.from_native(empty).select(nw.col("a").n_unique()).collect()["a"].to_list())  # [None]  expected [0]
print(pl.DataFrame({"a": []}, schema={"a": pl.Float64}).select(pl.col("a").n_unique())["a"].to_list())  # [0]
# Generated SQL: count(DISTINCT a) + max(CASE WHEN a IS NULL THEN 1 ELSE 0 END)  -> max() over no rows is NULL
```

## Output

```
[None]
[0]
```

## Expected behavior

`0`, as in Polars, pandas and PyArrow.

## Notes

- Wrapping the second term in `coalesce(..., 0)` (or using `count(*) > count(x)`) fixes it. The same expression is used for `n_unique().over(...)`.

## Versions

polars 1.44.2, pandas 3.0.6, pyarrow 25.0.1, duckdb 1.5.6, narwhals 2.26.0, Python 3.13.15

---
Found with differential fuzzing (Polars/pandas/DuckDB/PyArrow via Narwhals) and minimised automatically.
