"""Turn a minimal case into code a maintainer can paste and run.

Narwhals mirrors the Polars API, so the same renderer produces both a
Narwhals script (runs every backend side by side) and a native Polars script,
with no Narwhals in sight, for filing upstream with Polars.
"""

from __future__ import annotations

import json
import math

from .ir import BOOL, FLOAT, INT, STR

TYPE_NAMES = {INT: "Int64", FLOAT: "Float64", STR: "String", BOOL: "Boolean"}
SQL_TYPES = {INT: "BIGINT", FLOAT: "DOUBLE", STR: "VARCHAR", BOOL: "BOOLEAN"}
BIN_SYMBOLS = {"add": "+", "sub": "-", "mul": "*", "eq": "==", "ne": "!=", "lt": "<",
               "le": "<=", "gt": ">", "ge": ">=", "and": "&", "or": "|"}


def py_value(v) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return 'float("nan")'
        if math.isinf(v):
            return 'float("inf")' if v > 0 else 'float("-inf")'
    return repr(v)


def expr_code(e: dict, ns: str) -> str:
    k = e["op"]
    if k == "col":
        return f'{ns}.col("{e["name"]}")'
    if k == "lit":
        return f"{ns}.lit({py_value(e['value'])}, dtype={ns}.{TYPE_NAMES[e['dtype']]})"
    if k == "bin":
        return f"({expr_code(e['l'], ns)} {BIN_SYMBOLS[e['fn']]} {expr_code(e['r'], ns)})"
    if k == "un":
        x = expr_code(e["x"], ns)
        return {
            "not": f"(~{x})", "neg": f"(-{x})", "abs": f"{x}.abs()",
            "is_null": f"{x}.is_null()", "is_not_null": f"(~{x}.is_null())",
            "is_nan": f"{x}.is_nan()",
        }[e["fn"]]
    if k == "fill_null":
        return f"{expr_code(e['x'], ns)}.fill_null({py_value(e['value'])})"
    if k == "cast":
        return f"{expr_code(e['x'], ns)}.cast({ns}.{TYPE_NAMES[e['to']]})"
    if k == "str":
        x = expr_code(e["x"], ns)
        fn = e["fn"]
        if fn == "slice":
            return f"{x}.str.slice({e['offset']}, {e['length']!r})"
        if fn == "contains":
            return f"{x}.str.contains({e['arg']!r}, literal=True)"
        if fn in ("starts_with", "ends_with"):
            return f"{x}.str.{fn}({e['arg']!r})"
        return f"{x}.str.{fn}()"
    if k == "when":
        return (f"{ns}.when({expr_code(e['cond'], ns)}).then({expr_code(e['then'], ns)})"
                f".otherwise({expr_code(e['otherwise'], ns)})")
    if k == "over":
        by = ", ".join(repr(c) for c in e["by"])
        return f"{expr_code(e['x'], ns)}.{e['agg']}().over({by})"
    raise ValueError(k)


def agg_code(a: dict, ns: str) -> str:
    if a["fn"] == "len":
        return f'{ns}.len().alias("{a["name"]}")'
    return f'{ns}.col("{a["col"]}").{a["fn"]}().alias("{a["name"]}")'


def op_code(op: dict, ns: str) -> str:
    k = op["op"]
    if k == "filter":
        return f".filter({expr_code(op['pred'], ns)})"
    if k == "with_columns":
        return f'.with_columns({expr_code(op["expr"], ns)}.alias("{op["name"]}"))'
    if k == "select":
        return f".select({op['cols']!r})"
    if k == "group_by":
        aggs = ", ".join(agg_code(a, ns) for a in op["aggs"])
        return f".group_by({op['keys']!r}).agg({aggs})"
    if k == "agg":
        return f".select({', '.join(agg_code(a, ns) for a in op['aggs'])})"
    if k == "join":
        if op["how"] == "cross":
            return '.join(right, how="cross")'
        return f".join(right, on={op['on']!r}, how={op['how']!r})"
    if k in ("sort", "sort_head"):
        s = f".sort({op['by']!r}, descending={op['descending']!r}, nulls_last={op['nulls_last']!r})"
        return s + (f".head({op['n']})" if k == "sort_head" else "")
    if k == "unique":
        return ".unique()"
    if k == "drop_nulls":
        return f".drop_nulls(subset={op['subset']!r})"
    raise ValueError(k)


def pipeline_code(pipeline: list[dict], ns: str, var: str = "df") -> str:
    body = "\n".join(f"    {op_code(op, ns)}" for op in pipeline)
    return f"(\n    {var}\n{body}\n)"


def data_code(table: dict, var: str) -> str:
    names = list(table["columns"])
    lines = [f"{var}_data = {{"]
    for i, n in enumerate(names):
        vals = ", ".join(py_value(r[i]) for r in table["rows"])
        lines.append(f'    "{n}": [{vals}],')
    lines.append("}")
    schema = ", ".join(f'"{n}": pl.{TYPE_NAMES[t]}' for n, t in table["columns"].items())
    lines.append(f"{var}_schema = {{{schema}}}")
    return "\n".join(lines)


def polars_script(case: dict, mode: str = "eager") -> str:
    """Native Polars reproducer. mode: eager | lazy | noopt | streaming | chunked."""
    parts = ["import polars as pl", "", data_code(case["tables"][0], "left")]
    has_right = len(case["tables"]) > 1
    if has_right:
        parts.append(data_code(case["tables"][1], "right"))
    parts.append("")
    parts.append("df = pl.DataFrame(left_data, schema=left_schema)")
    if mode == "chunked":
        parts.append("df = pl.concat([df[: df.height // 2], df[df.height // 2 :]], rechunk=False)")
    if has_right:
        parts.append("right = pl.DataFrame(right_data, schema=right_schema)")
    lazy = mode in ("lazy", "noopt", "streaming")
    if lazy:
        parts.append("df = df.lazy()")
        if has_right:
            parts.append("right = right.lazy()")
    parts.append(f"q = {pipeline_code(case['pipeline'], 'pl')}")
    if mode == "lazy":
        parts.append("print(q.collect())")
    elif mode == "noopt":
        parts.append("print(q.collect())  # default optimizations")
        parts.append("print(q.collect(optimizations=pl.QueryOptFlags.none()))  # optimizer off")
    elif mode == "streaming":
        parts.append('print(q.collect())  # in-memory engine')
        parts.append('print(q.collect(engine="streaming"))  # streaming engine')
    else:
        parts.append("print(q)")
    return "\n".join(parts) + "\n"


def narwhals_script(case: dict) -> str:
    """Runs the case on every backend through Narwhals, printing each result."""
    from .engines import BACKEND_NAMES  # local import keeps render importable alone

    parts = [
        "import json",
        "from framebreaker.engines import make_backends, run_case",
        "from framebreaker.compare import show_rows",
        "",
        f"CASE = json.loads({_json_literal(case)})",
        "",
        "for backend in make_backends():",
        '    print(f"--- {backend.name}")',
        "    print(show_rows(run_case(CASE, backend)))",
        "",
        "# Equivalent Narwhals pipeline:",
    ]
    parts += ["# " + line for line in pipeline_code(case["pipeline"], "nw").splitlines()]
    parts.append(f"# backends: {', '.join(BACKEND_NAMES)}")
    return "\n".join(parts) + "\n"


def _json_literal(case: dict) -> str:
    return repr(json.dumps(case))


def sql_value(v, dtype: str) -> str:
    if v is None:
        return f"NULL::{SQL_TYPES[dtype]}"
    if dtype == FLOAT:
        if math.isnan(v):
            return "'nan'::DOUBLE"
        if math.isinf(v):
            return "'inf'::DOUBLE" if v > 0 else "'-inf'::DOUBLE"
        return f"{v!r}::DOUBLE"
    if dtype == STR:
        return "'" + v.replace("'", "''") + "'"
    if dtype == BOOL:
        return "true" if v else "false"
    return str(v)


def duckdb_sql(case: dict) -> str | None:
    """SQL reproducer using the query Narwhals generates for DuckDB."""
    import narwhals as nw

    from .engines import make_backends, run_pipeline, to_arrow

    duck = make_backends(["duckdb"])[0]
    try:
        frames = [duck.load(to_arrow(t), i) for i, t in enumerate(case["tables"])]
        right = frames[1] if len(frames) > 1 else None
        rel = run_pipeline(frames[0], case["pipeline"], right)
        query = nw.to_native(rel).sql_query()
    except Exception:
        return None
    stmts = []
    for i, t in enumerate(case["tables"]):
        cols = ", ".join(f"{c} {SQL_TYPES[d]}" for c, d in t["columns"].items())
        stmts.append(f"CREATE TABLE t{i} ({cols});")
        if t["rows"]:
            dtypes = list(t["columns"].values())
            rows = ", ".join("(" + ", ".join(sql_value(v, d) for v, d in zip(r, dtypes)) + ")"
                             for r in t["rows"])
            stmts.append(f"INSERT INTO t{i} VALUES {rows};")
    stmts.append(query + ";")
    return "\n".join(stmts) + "\n"
