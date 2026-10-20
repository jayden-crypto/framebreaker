<!-- file at https://github.com/pandas-dev/pandas/issues/new -->
# BUG: merge on numpy bool vs bool[pyarrow] key containing NA raises ArrowInvalid

## Description

Merging a numpy `bool` key with a `bool[pyarrow]` key that contains a missing value raises `ArrowInvalid: Could not convert 1 with type int: tried to convert to boolean`. Without the NA, or with `bool[pyarrow]` on both sides, the merge works.

## Reproducible example

```python
import pandas as pd
import pyarrow as pa

left = pd.DataFrame({"k": [False]})  # numpy bool
right = pd.DataFrame({"k": pd.array([None], dtype=pd.ArrowDtype(pa.bool_()))})

print(left.merge(pd.DataFrame({"k": pd.array([True], dtype=pd.ArrowDtype(pa.bool_()))}), on="k"))  # fine, empty
left.merge(right, on="k")
# pyarrow.lib.ArrowInvalid: Could not convert 1 with type int: tried to convert to boolean
```

## Output

```
Empty DataFrame
Columns: [k]
Index: []
pyarrow.lib.ArrowInvalid: Could not convert 1 with type int: tried to convert to boolean
```

## Expected behavior

The merge succeeds (the NA key simply doesn't match), returning an empty frame here.

## Notes

- Traceback goes through `merge._factorize_keys`, which for `ArrowExtensionArray` calls `lk.to_numpy(na_value=1, dtype=lk.dtype.numpy_dtype)` under the comment "we can only get here with numeric dtypes". Boolean Arrow arrays do reach it, and `1` can't be boxed as an Arrow boolean. Using a dtype-appropriate placeholder (e.g. `True` for bool) would fix it.

## Versions

polars 1.44.2, pandas 3.0.6, pyarrow 25.0.1, duckdb 1.5.6, narwhals 2.26.0, Python 3.13.15

---
Found with differential fuzzing (Polars/pandas/DuckDB/PyArrow via Narwhals) and minimised automatically.
