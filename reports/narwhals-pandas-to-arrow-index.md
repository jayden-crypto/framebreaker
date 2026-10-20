<!-- file at https://github.com/narwhals-dev/narwhals/issues/new -->
# [Bug]: `DataFrame.to_arrow()` on pandas leaks the index as `__index_level_0__` after a filter

## Description

For pandas-backed frames, `to_arrow()` calls `pa.Table.from_pandas(self.native)` without `preserve_index=False`. Whether an extra `__index_level_0__` column appears then depends on *which rows* a previous filter kept: a contiguous index is stored as metadata, a non-contiguous one becomes a column. The output schema depends on the data, and doesn't match other backends.

## Reproducible example

```python
import narwhals as nw
import pandas as pd

df = nw.from_native(pd.DataFrame({"a": [0, 1, 0, 0]}), eager_only=True)

print(df.filter(nw.col("a") > 0).to_arrow().column_names)   # ['a']
print(df.filter(nw.col("a") == 0).to_arrow().column_names)  # ['a', '__index_level_0__']
```

## Output

```
['a']
['a', '__index_level_0__']
```

## Expected behavior

`to_arrow()` returns only the frame's columns, as for every other backend.

## Notes

- One-line fix: `pa.Table.from_pandas(self.native, preserve_index=False)` (the cuDF branch already does this). Still present on `main`.

## Versions

polars 1.44.2, pandas 3.0.6, pyarrow 25.0.1, duckdb 1.5.6, narwhals 2.26.0, Python 3.13.15

---
Found with differential fuzzing (Polars/pandas/DuckDB/PyArrow via Narwhals) and minimised automatically.
