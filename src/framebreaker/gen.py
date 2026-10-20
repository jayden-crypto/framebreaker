"""Random case generation.

Values are drawn from small pools of "nasty" inputs (nulls, NaN, -0.0, empty
strings, combining characters...) because small tables full of edge cases find
far more bugs than large tables of ordinary values. Floats in the pool are
dyadic rationals, so sums are exact in any order and engines can be compared
without loose tolerances.
"""

from __future__ import annotations

import random

from .ir import (
    AGGS,
    ARITH,
    BOOL,
    CMP,
    DTYPES,
    FLOAT,
    INT,
    JOIN_HOWS,
    NUMERIC,
    OP_KINDS,
    STR,
    STR_TO_BOOL,
    STR_TO_STR,
    WINDOW_AGGS,
    InvalidCase,
    Schema,
    apply_schema,
    validate,
)

INT_POOL = [0, 1, -1, 2, 3, -7, 42, 100, -1000, 9999]
FLOAT_POOL = [0.0, -0.0, 0.5, -1.25, 3.0, 2.5, -3.75, 1000000.25]
FLOAT_SPECIAL = [float("nan"), float("inf"), float("-inf")]
STR_POOL = ["", "a", "A", "b", "ab", "abc", " a", "a ", "ß", "é", "é",
            "😀", "日本", "İ", "NULL", "0", "a%b", "a_b"]
STR_ARGS = ["", "a", "b", "A", "é", " ", "%", "_", "😀"]


def gen_value(rng: random.Random, dtype: str, null_p: float = 0.0, special_p: float = 0.1):
    if rng.random() < null_p:
        return None
    if dtype == INT:
        return rng.choice(INT_POOL)
    if dtype == FLOAT:
        if rng.random() < special_p:
            return rng.choice(FLOAT_SPECIAL)
        return rng.choice(FLOAT_POOL)
    if dtype == STR:
        return rng.choice(STR_POOL)
    return rng.choice([True, False])


def gen_table(rng: random.Random, columns: dict[str, str], n_rows: int) -> dict:
    null_p = {c: rng.choice([0.0, 0.15, 0.4]) for c in columns}
    base = [[gen_value(rng, t, null_p[c]) for c, t in columns.items()]
            for _ in range(min(n_rows, 40))]
    rows = base if n_rows <= 40 else [list(rng.choice(base)) for _ in range(n_rows)]
    return {"columns": dict(columns), "rows": rows}


def gen_row_count(rng: random.Random) -> int:
    r = rng.random()
    if r < 0.05:
        return 0
    if r < 0.80:
        return rng.randint(1, 8)
    if r < 0.95:
        return rng.randint(9, 40)
    return rng.randint(200, 3000)


class Generator:
    def __init__(self, rng: random.Random, weights: dict[str, float] | None = None,
                 max_ops: int = 5):
        self.rng = rng
        self.weights = weights or {k: 1.0 for k in OP_KINDS}
        self.max_ops = max_ops

    # ---------------------------------------------------------------- tables
    def gen_tables(self) -> list[dict]:
        rng = self.rng
        n_cols = rng.randint(2, 5)
        cols = {f"c{i}": rng.choice(DTYPES) for i in range(n_cols)}
        t0 = gen_table(rng, cols, gen_row_count(rng))
        if rng.random() < 0.5:
            return [t0]
        shared = rng.sample(list(cols), k=rng.randint(1, min(2, n_cols)))
        rcols = {c: cols[c] for c in shared}
        for i in range(rng.randint(1, 2)):
            rcols[f"r{i}"] = rng.choice(DTYPES)
        t1 = gen_table(rng, rcols, rng.choice([0, 1, 2, 3, 5, 8]))
        return [t0, t1]

    # ----------------------------------------------------------- expressions
    def lit(self, dtype: str) -> dict:
        v = gen_value(self.rng, dtype, null_p=0.08, special_p=0.05)
        return {"op": "lit", "value": v, "dtype": dtype}

    def leaf(self, schema: Schema, dtype: str) -> dict:
        cols = [c for c, t in schema.items() if t == dtype]
        if cols and self.rng.random() < 0.75:
            return {"op": "col", "name": self.rng.choice(cols)}
        return self.lit(dtype)

    def expr(self, schema: Schema, dtype: str, depth: int = 2) -> dict:
        rng = self.rng
        if depth <= 0 or rng.random() < 0.3:
            return self.leaf(schema, dtype)
        d = depth - 1
        choices = ["when", "fill_null"]
        if dtype in NUMERIC:
            choices += ["arith", "arith", "neg_abs", "over"]
        if dtype == INT:
            choices += ["len_chars", "cast_bool"]
        if dtype == FLOAT:
            choices += ["cast_int_float"]
        if dtype == STR:
            choices += ["str_fn", "str_fn", "cast_int_str"]
        if dtype == BOOL:
            choices += ["cmp", "cmp", "logic", "not", "nullcheck", "is_nan", "str_pred"]
        kind = rng.choice(choices)

        if kind == "arith":
            if dtype == INT:
                lt = rt = INT
            else:
                lt, rt = rng.choice([(FLOAT, FLOAT), (INT, FLOAT), (FLOAT, INT)])
            return {"op": "bin", "fn": rng.choice(ARITH),
                    "l": self.expr(schema, lt, d), "r": self.expr(schema, rt, d)}
        if kind == "neg_abs":
            return {"op": "un", "fn": rng.choice(["neg", "abs"]), "x": self.expr(schema, dtype, d)}
        if kind == "over":
            keys = [c for c in schema if schema[c] != FLOAT] or list(schema)
            # pick an aggregation whose output dtype is the one we need
            options = [(f, dtype) for f in ("sum", "min", "max")]
            if dtype == INT:
                options += [("count", t) for t in DTYPES]
            else:
                options += [("mean", t) for t in NUMERIC]
            fn, in_t = rng.choice(options)
            assert fn in WINDOW_AGGS
            return {"op": "over", "agg": fn, "x": self.leaf(schema, in_t),
                    "by": rng.sample(keys, k=1)}
        if kind == "len_chars":
            return {"op": "str", "fn": "len_chars", "x": self.expr(schema, STR, d)}
        if kind == "cast_bool":
            return {"op": "cast", "to": INT, "x": self.expr(schema, BOOL, d)}
        if kind == "cast_int_float":
            return {"op": "cast", "to": FLOAT, "x": self.expr(schema, INT, d)}
        if kind == "cast_int_str":
            return {"op": "cast", "to": STR, "x": self.expr(schema, INT, d)}
        if kind == "str_fn":
            fn = rng.choice(STR_TO_STR)
            e = {"op": "str", "fn": fn, "x": self.expr(schema, STR, d)}
            if fn == "slice":
                e["offset"] = rng.randint(-3, 3)
                e["length"] = rng.choice([None, 0, 1, 2, 3])
            return e
        if kind == "str_pred":
            return {"op": "str", "fn": rng.choice(STR_TO_BOOL),
                    "x": self.expr(schema, STR, d), "arg": rng.choice(STR_ARGS)}
        if kind == "cmp":
            t = rng.choice(DTYPES)
            fn = rng.choice(CMP)
            return {"op": "bin", "fn": fn, "l": self.expr(schema, t, d), "r": self.expr(schema, t, d)}
        if kind == "logic":
            return {"op": "bin", "fn": rng.choice(["and", "or"]),
                    "l": self.expr(schema, BOOL, d), "r": self.expr(schema, BOOL, d)}
        if kind == "not":
            return {"op": "un", "fn": "not", "x": self.expr(schema, BOOL, d)}
        if kind == "nullcheck":
            return {"op": "un", "fn": rng.choice(["is_null", "is_not_null"]),
                    "x": self.expr(schema, rng.choice(DTYPES), d)}
        if kind == "is_nan":
            return {"op": "un", "fn": "is_nan", "x": self.expr(schema, FLOAT, d)}
        if kind == "when":
            return {"op": "when", "cond": self.expr(schema, BOOL, d),
                    "then": self.expr(schema, dtype, d), "otherwise": self.expr(schema, dtype, d)}
        # fill_null
        v = gen_value(rng, dtype, special_p=0.0)
        return {"op": "fill_null", "x": self.expr(schema, dtype, d), "value": v, "dtype": dtype}

    # ------------------------------------------------------------ operations
    def aggs(self, schema: Schema, avoid: set[str]) -> list[dict]:
        out = []
        for i in range(self.rng.randint(1, 3)):
            fn = self.rng.choice(list(AGGS))
            name = f"a{i}"
            while name in avoid:
                name += "_"
            if fn == "len":
                out.append({"name": name, "fn": "len", "col": None})
                continue
            cols = [c for c, t in schema.items() if t in AGGS[fn][0]]
            if not cols:
                out.append({"name": name, "fn": "len", "col": None})
                continue
            out.append({"name": name, "fn": fn, "col": self.rng.choice(cols)})
        return out

    def op(self, kind: str, schema: Schema, right: Schema | None, step: int) -> dict:
        rng = self.rng
        cols = list(schema)
        if kind == "filter":
            return {"op": "filter", "pred": self.expr(schema, BOOL, rng.randint(1, 3))}
        if kind == "with_columns":
            if rng.random() < 0.25:
                name = rng.choice(cols)
                dtype = rng.choice([schema[name], rng.choice(DTYPES)])
            else:
                name, dtype = f"e{step}", rng.choice(DTYPES)
            return {"op": "with_columns", "name": name, "expr": self.expr(schema, dtype, rng.randint(1, 3))}
        if kind == "select":
            k = rng.randint(1, len(cols))
            return {"op": "select", "cols": rng.sample(cols, k)}
        if kind == "group_by":
            keys = rng.sample(cols, k=rng.randint(1, min(2, len(cols))))
            return {"op": "group_by", "keys": keys, "aggs": self.aggs(schema, set(keys))}
        if kind == "agg":
            return {"op": "agg", "aggs": self.aggs(schema, set())}
        if kind == "join":
            how = rng.choice(JOIN_HOWS)
            if how == "cross":
                return {"op": "join", "how": how, "on": []}
            keys = [c for c in right if c in schema and schema[c] == right[c]]
            return {"op": "join", "how": how, "on": rng.sample(keys, k=rng.randint(1, len(keys)))}
        if kind == "sort":
            by = rng.sample(cols, k=rng.randint(1, min(3, len(cols))))
            return {"op": "sort", "by": by, "descending": [rng.random() < 0.5 for _ in by],
                    "nulls_last": rng.random() < 0.5}
        if kind == "sort_head":
            by = rng.sample(cols, k=len(cols))
            return {"op": "sort_head", "by": by, "descending": [rng.random() < 0.5 for _ in by],
                    "nulls_last": rng.random() < 0.5, "n": rng.randint(0, 5)}
        if kind == "unique":
            return {"op": "unique"}
        subset = None if rng.random() < 0.4 else rng.sample(cols, k=rng.randint(1, len(cols)))
        return {"op": "drop_nulls", "subset": subset}

    def applicable(self, schema: Schema, right: Schema | None, joined: bool) -> list[str]:
        kinds = [k for k in OP_KINDS if k != "join"]
        if right is not None and not joined:
            if any(c in schema and schema[c] == right[c] for c in right):
                kinds.append("join")
        return kinds

    def gen_case(self) -> dict:
        for _ in range(100):
            tables = self.gen_tables()
            schema = dict(tables[0]["columns"])
            right = dict(tables[1]["columns"]) if len(tables) > 1 else None
            pipeline: list[dict] = []
            joined = False
            for step in range(self.rng.randint(1, self.max_ops)):
                kinds = self.applicable(schema, right, joined)
                kind = self.rng.choices(kinds, weights=[self.weights.get(k, 1.0) for k in kinds])[0]
                op = self.op(kind, schema, right, step)
                try:
                    schema = apply_schema(op, schema, right)
                except InvalidCase:
                    continue
                pipeline.append(op)
                joined = joined or kind == "join"
            if not pipeline:
                continue
            case = {"tables": tables, "pipeline": pipeline}
            try:
                validate(case)
            except InvalidCase:
                continue
            return case
        raise RuntimeError("could not generate a valid case")
