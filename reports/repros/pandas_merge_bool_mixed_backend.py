import pandas as pd
import pyarrow as pa

left = pd.DataFrame({"k": [False]})  # numpy bool
right = pd.DataFrame({"k": pd.array([None], dtype=pd.ArrowDtype(pa.bool_()))})

print(left.merge(pd.DataFrame({"k": pd.array([True], dtype=pd.ArrowDtype(pa.bool_()))}), on="k"))  # fine, empty
left.merge(right, on="k")
# pyarrow.lib.ArrowInvalid: Could not convert 1 with type int: tried to convert to boolean
