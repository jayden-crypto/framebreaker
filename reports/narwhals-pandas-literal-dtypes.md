<!-- file at https://github.com/narwhals-dev/narwhals/issues/new -->
# [Bug]: pandas-like `lit()` mishandles nulls and dtypes (`lit(None, Boolean)` becomes False)

## Description

On the pandas backend, literals are built with `from_iterable([value])` and then cast, which goes through numpy dtypes regardless of the frame's dtype backend:

1. `nw.lit(None, dtype=nw.Boolean)` silently becomes `False` (wrong data).
2. `nw.lit(None, dtype=nw.Int64)` raises `TypeError`.
3. On pyarrow-backed frames, a string literal in `when/then/otherwise` produces pandas 3's NaN-semantics `str` dtype, so null propagation is lost (`str.starts_with` on a null returns `False` instead of null).

## Reproducible example

```python
import narwhals as nw
import pandas as pd

df = nw.from_native(pd.DataFrame({"a": [1, 2]}), eager_only=True)

# 1. Null boolean literal silently becomes False
print(df.with_columns(b=nw.lit(None, dtype=nw.Boolean))["b"].to_list())   # [False, False]  expected [None, None]

# 2. Null integer literal raises
try:
    df.with_columns(b=nw.lit(None, dtype=nw.Int64))
except TypeError as e:
    print("TypeError:", e)  # int() argument must be ... not 'NoneType'

# 3. On a pyarrow-backed frame, a string literal gets pandas 3's NaN-semantics `str`
#    dtype, so null-propagation is lost:
pdf = pd.DataFrame({"flag": pd.array([False], dtype="bool[pyarrow]"),
                    "s": pd.array([None], dtype="string[pyarrow]")})
f = nw.from_native(pdf, eager_only=True)
e = nw.when(nw.col("flag")).then(nw.lit("")).otherwise(nw.col("s"))
print(f.select(x=e).to_native().dtypes["x"])                   # str (na_value=nan)
print(f.select(x=e.str.starts_with(" "))["x"].to_list())       # [False]  expected [None] (as in Polars)
```

## Output

```
[False, False]
TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'
str
[False]
```

## Expected behavior

Polars semantics: typed null literals stay null, and literal dtypes follow the frame's backend.

## Notes

- 1 and 2 also happen on numpy-backed frames. 3 also makes joins against `bool[pyarrow]` keys fail (pandas then hits a separate merge bug).

## Versions

polars 1.44.2, pandas 3.0.6, pyarrow 25.0.1, duckdb 1.5.6, narwhals 2.26.0, Python 3.13.15

---
Found with differential fuzzing (Polars/pandas/DuckDB/PyArrow via Narwhals) and minimised automatically.
