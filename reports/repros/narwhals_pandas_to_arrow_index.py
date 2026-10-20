import narwhals as nw
import pandas as pd

df = nw.from_native(pd.DataFrame({"a": [0, 1, 0, 0]}), eager_only=True)

print(df.filter(nw.col("a") > 0).to_arrow().column_names)   # ['a']
print(df.filter(nw.col("a") == 0).to_arrow().column_names)  # ['a', '__index_level_0__']
