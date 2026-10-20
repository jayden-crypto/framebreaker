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
