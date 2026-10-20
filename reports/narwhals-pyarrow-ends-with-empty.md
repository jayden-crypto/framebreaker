<!-- file at https://github.com/narwhals-dev/narwhals/issues/new -->
# [Bug]: `str.ends_with('')` returns False for non-empty strings on PyArrow

## Description

Every string ends with the empty string, and PyArrow's own `pc.ends_with(..., pattern='')` returns True. Narwhals' PyArrow backend compares `utf8_slice_codeunits(s, start=-len(suffix))` with the suffix; for an empty suffix `-0 == 0` slices the *whole* string, which only equals `''` for empty strings.

## Reproducible example

```python
import narwhals as nw
import pyarrow as pa
import pyarrow.compute as pc

df = nw.from_native(pa.table({"c": ["ab", "", "x"]}), eager_only=True)
print(df.select(nw.col("c").str.ends_with(""))["c"].to_list())    # [False, True, False]  expected all True
print(df.select(nw.col("c").str.starts_with(""))["c"].to_list())  # [True, True, True]
print(pc.ends_with(pa.array(["ab", "", "x"]), pattern="").to_pylist())  # [True, True, True]
# narwhals compares utf8_slice_codeunits(s, start=-len(suffix)) with the suffix; -0 == 0 slices the whole string.
```

## Output

```
[False, True, False]
[True, True, True]
[True, True, True]
```

## Expected behavior

`[True, True, True]`, as for `starts_with('')` and the other backends.

## Notes

- Using `pc.ends_with(native, pattern=suffix)` directly would fix it.

## Versions

polars 1.44.2, pandas 3.0.6, pyarrow 25.0.1, duckdb 1.5.6, narwhals 2.26.0, Python 3.13.15

---
Found with differential fuzzing (Polars/pandas/DuckDB/PyArrow via Narwhals) and minimised automatically.
