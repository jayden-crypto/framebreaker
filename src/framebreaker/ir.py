"""Test-case IR.

A case is a JSON-friendly dict:

    {
      "tables": [{"columns": {"c0": "int", ...}, "rows": [[1, ...], ...]}, ...],
      "pipeline": [{"op": "filter", "pred": <expr>}, ...],
    }

Table 0 is the input; table 1 (optional) is the right side of joins.
Expressions are nested dicts such as {"op": "bin", "fn": "add", "l": ..., "r": ...}.
Keeping everything as plain data lets us generate, serialise, shrink and render
cases without caring which engine will eventually run them.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Iterator

INT, FLOAT, STR, BOOL = "int", "float", "str", "bool"
DTYPES = (INT, FLOAT, STR, BOOL)
NUMERIC = (INT, FLOAT)

ARITH = ("add", "sub", "mul")
CMP = ("eq", "ne", "lt", "le", "gt", "ge")
LOGIC = ("and", "or")
STR_TO_STR = ("to_uppercase", "to_lowercase", "strip_chars", "slice")
STR_TO_BOOL = ("contains", "starts_with", "ends_with")
STR_TO_INT = ("len_chars",)
CASTS = {(INT, FLOAT), (BOOL, INT), (INT, STR)}

# aggregation -> (accepted input dtypes, output dtype or None for "same as input")
AGGS: dict[str, tuple[tuple[str, ...], str | None]] = {
    "sum": (NUMERIC, None),
    "mean": (NUMERIC, FLOAT),
    "median": (NUMERIC, FLOAT),
    "std": (NUMERIC, FLOAT),
    "min": ((INT, FLOAT, STR), None),
    "max": ((INT, FLOAT, STR), None),
    "count": (DTYPES, INT),
    "n_unique": (DTYPES, INT),
    "len": ((), INT),  # takes no column
}
WINDOW_AGGS = ("sum", "min", "max", "count", "mean")

JOIN_HOWS = ("inner", "left", "full", "semi", "anti", "cross")
OP_KINDS = (
    "filter", "with_columns", "select", "group_by", "agg", "join",
    "sort", "sort_head", "unique", "drop_nulls",
)


class InvalidCase(Exception):
    """Raised when a (usually shrunk) case is not well-typed."""


Schema = dict[str, str]


def expr_dtype(e: dict, schema: Schema) -> str:
    k = e["op"]
    if k == "col":
        if e["name"] not in schema:
            raise InvalidCase(f"unknown column {e['name']}")
        return schema[e["name"]]
    if k == "lit":
        return e["dtype"]
    if k == "bin":
        lt, rt = expr_dtype(e["l"], schema), expr_dtype(e["r"], schema)
        fn = e["fn"]
        if fn in ARITH:
            if lt not in NUMERIC or rt not in NUMERIC:
                raise InvalidCase("arith on non-numeric")
            return INT if lt == rt == INT else FLOAT
        if fn in CMP:
            if not (lt == rt or (lt in NUMERIC and rt in NUMERIC)):
                raise InvalidCase("comparison of incompatible types")
            return BOOL
        if fn in LOGIC:
            if lt != BOOL or rt != BOOL:
                raise InvalidCase("logic on non-bool")
            return BOOL
        raise InvalidCase(f"unknown binary fn {fn}")
    if k == "un":
        t = expr_dtype(e["x"], schema)
        fn = e["fn"]
        if fn == "not":
            if t != BOOL:
                raise InvalidCase("not on non-bool")
            return BOOL
        if fn in ("neg", "abs"):
            if t not in NUMERIC:
                raise InvalidCase("neg/abs on non-numeric")
            return t
        if fn in ("is_null", "is_not_null"):
            return BOOL
        if fn == "is_nan":
            if t != FLOAT:
                raise InvalidCase("is_nan on non-float")
            return BOOL
        raise InvalidCase(f"unknown unary fn {fn}")
    if k == "fill_null":
        t = expr_dtype(e["x"], schema)
        if t != e["dtype"]:
            raise InvalidCase("fill_null dtype mismatch")
        return t
    if k == "cast":
        t = expr_dtype(e["x"], schema)
        if (t, e["to"]) not in CASTS:
            raise InvalidCase("unsupported cast")
        return e["to"]
    if k == "str":
        if expr_dtype(e["x"], schema) != STR:
            raise InvalidCase("str fn on non-str")
        fn = e["fn"]
        if fn in STR_TO_STR:
            return STR
        if fn in STR_TO_BOOL:
            return BOOL
        if fn in STR_TO_INT:
            return INT
        raise InvalidCase(f"unknown str fn {fn}")
    if k == "when":
        if expr_dtype(e["cond"], schema) != BOOL:
            raise InvalidCase("when condition not bool")
        a, b = expr_dtype(e["then"], schema), expr_dtype(e["otherwise"], schema)
        if a != b:
            raise InvalidCase("when branches differ")
        return a
    if k == "over":
        for c in e["by"]:
            if c not in schema:
                raise InvalidCase(f"unknown partition column {c}")
        return agg_dtype(e["agg"], expr_dtype(e["x"], schema))
    raise InvalidCase(f"unknown expr {k}")


def agg_dtype(fn: str, in_dtype: str | None) -> str:
    accepted, out = AGGS[fn]
    if fn == "len":
        return INT
    if in_dtype not in accepted:
        raise InvalidCase(f"{fn} does not accept {in_dtype}")
    return out or in_dtype


def table_schema(table: dict) -> Schema:
    return dict(table["columns"])


def apply_schema(op: dict, schema: Schema, right: Schema | None) -> Schema:
    """Return the schema after `op`, or raise InvalidCase."""
    k = op["op"]
    if k == "filter":
        if expr_dtype(op["pred"], schema) != BOOL:
            raise InvalidCase("filter predicate not bool")
        return schema
    if k == "with_columns":
        out = dict(schema)
        out[op["name"]] = expr_dtype(op["expr"], schema)
        return out
    if k == "select":
        if not op["cols"] or len(set(op["cols"])) != len(op["cols"]):
            raise InvalidCase("bad select")
        for c in op["cols"]:
            if c not in schema:
                raise InvalidCase(f"unknown column {c}")
        return {c: schema[c] for c in op["cols"]}
    if k in ("group_by", "agg"):
        keys = op.get("keys", [])
        if k == "group_by" and not keys:
            raise InvalidCase("group_by without keys")
        if not op["aggs"]:
            raise InvalidCase("no aggregations")
        out = {}
        for c in keys:
            if c not in schema or c in out:
                raise InvalidCase("bad group key")
            out[c] = schema[c]
        for a in op["aggs"]:
            if a["name"] in out:
                raise InvalidCase("duplicate output name")
            if a["fn"] != "len" and a["col"] not in schema:
                raise InvalidCase("unknown agg column")
            out[a["name"]] = agg_dtype(a["fn"], schema.get(a.get("col")))
        return out
    if k == "join":
        if right is None:
            raise InvalidCase("join without right table")
        how = op["how"]
        on = op.get("on", [])
        if how != "cross":
            if not on:
                raise InvalidCase("join without keys")
            for c in on:
                if c not in schema or c not in right or schema[c] != right[c]:
                    raise InvalidCase("bad join key")
        if how in ("semi", "anti"):
            return schema
        out = dict(schema)
        coalesced = set(on) if how in ("inner", "left") else set()
        for c, t in right.items():
            if c in coalesced:
                continue
            name = c if c not in out else c + "_right"
            if name in out:
                raise InvalidCase("join output name clash")
            out[name] = t
        return out
    if k in ("sort", "sort_head"):
        if not op["by"] or len(op["by"]) != len(op["descending"]):
            raise InvalidCase("bad sort")
        for c in op["by"]:
            if c not in schema:
                raise InvalidCase(f"unknown sort column {c}")
        if k == "sort_head" and set(op["by"]) != set(schema):
            # sorting by every column makes head() deterministic
            raise InvalidCase("sort_head must sort by all columns")
        return schema
    if k == "unique":
        return schema
    if k == "drop_nulls":
        for c in op.get("subset") or []:
            if c not in schema:
                raise InvalidCase(f"unknown column {c}")
        return schema
    raise InvalidCase(f"unknown op {k}")


def validate(case: dict) -> Schema:
    """Type-check a whole case and return its output schema."""
    tables = case["tables"]
    for t in tables:
        n = len(t["columns"])
        if n == 0 or any(len(r) != n for r in t["rows"]):
            raise InvalidCase("ragged or empty table")
    schema = table_schema(tables[0])
    right = table_schema(tables[1]) if len(tables) > 1 else None
    for op in case["pipeline"]:
        schema = apply_schema(op, schema, right)
    return schema


def is_ordered(case: dict) -> bool:
    """True when the final row order is fully determined by the pipeline."""
    return bool(case["pipeline"]) and case["pipeline"][-1]["op"] == "sort_head"


def iter_exprs(node: Any) -> Iterator[dict]:
    """Yield every expression dict nested anywhere inside `node`."""
    if isinstance(node, dict):
        if "op" in node and node["op"] in EXPR_KINDS:
            yield node
        for v in node.values():
            yield from iter_exprs(v)
    elif isinstance(node, list):
        for v in node:
            yield from iter_exprs(v)


EXPR_KINDS = {"col", "lit", "bin", "un", "fill_null", "cast", "str", "when", "over"}


def features(case: dict) -> list[str]:
    """Coarse labels describing what a case exercises (used for signatures)."""
    feats = set()
    for op in case["pipeline"]:
        feats.add(op["op"] if op["op"] != "join" else f"join:{op['how']}")
        for a in op.get("aggs", []):
            feats.add(f"agg:{a['fn']}")
    for e in iter_exprs(case["pipeline"]):
        if e["op"] in ("bin", "un", "str"):
            feats.add(f"{e['op']}:{e['fn']}")
        elif e["op"] == "over":
            feats.add(f"over:{e['agg']}")
        elif e["op"] in ("when", "fill_null", "cast"):
            feats.add(e["op"])
    return sorted(feats)


def _value_tags(v, tags: set) -> None:
    if v is None:
        tags.add("null")
    elif isinstance(v, float):
        if v != v:
            tags.add("nan")
        elif v in (float("inf"), float("-inf")):
            tags.add("inf")
        elif v == 0 and str(v).startswith("-"):
            tags.add("neg-zero")
    elif isinstance(v, str) and not v.isascii():
        tags.add("unicode")


def data_tags(case: dict) -> list[str]:
    """Labels for nasty values in the data *or* literals of a case, to speed up triage."""
    tags: set[str] = set()
    for t in case["tables"]:
        if not t["rows"]:
            tags.add("empty-table")
        for row in t["rows"]:
            for v in row:
                _value_tags(v, tags)
    for e in iter_exprs([case["pipeline"], case.get("tlp")]):
        if e["op"] == "lit":
            _value_tags(e["value"], tags)
        elif e["op"] == "fill_null":
            _value_tags(e["value"], tags)
        elif e["op"] == "str" and isinstance(e.get("arg"), str):
            _value_tags(e["arg"], tags)
    return sorted(tags)


def clone(case: dict) -> dict:
    return copy.deepcopy(case)


def dumps(case: dict) -> str:
    return json.dumps(case, indent=1)
