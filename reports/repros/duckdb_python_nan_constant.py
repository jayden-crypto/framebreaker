import duckdb

rel = duckdb.sql("SELECT 1 AS a")
print(duckdb.sql("SELECT 'nan'::DOUBLE AS x").fetchall())                        # [(nan,)]
print(rel.select(duckdb.ConstantExpression(float("nan")).alias("x")).fetchall())  # [(None,)]  expected [(nan,)]
print(rel.select(duckdb.ConstantExpression(float("inf")).alias("x")).fetchall())  # [(inf,)]
