import narwhals as nw
import pyarrow as pa
import pyarrow.compute as pc

df = nw.from_native(pa.table({"c": ["ab", "", "x"]}), eager_only=True)
print(df.select(nw.col("c").str.ends_with(""))["c"].to_list())    # [False, True, False]  expected all True
print(df.select(nw.col("c").str.starts_with(""))["c"].to_list())  # [True, True, True]
print(pc.ends_with(pa.array(["ab", "", "x"]), pattern="").to_pylist())  # [True, True, True]
# narwhals compares utf8_slice_codeunits(s, start=-len(suffix)) with the suffix; -0 == 0 slices the whole string.
