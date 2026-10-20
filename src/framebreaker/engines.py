"""Execute a case on many engines through one Narwhals interpreter.

Every backend gets the same pipeline. Polars is run several ways (eager, lazy,
lazy with the optimizer disabled, the streaming engine, multi-chunk input) so
that disagreements between those variants point straight at optimizer or
engine-specific bugs, with no second library needed.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, Callable

import duckdb
import narwhals as nw
import pandas as pd
import polars as pl
import polars.exceptions as ple
import pyarrow as pa

from .compare import normalize
from .ir import BOOL, FLOAT, INT, STR, is_ordered

ARROW_TYPES = {INT: pa.int64(), FLOAT: pa.float64(), STR: pa.string(), BOOL: pa.bool_()}
NW_TYPES = {INT: nw.Int64, FLOAT: nw.Float64, STR: nw.String, BOOL: nw.Boolean}

BINARY: dict[str, Callable[[Any, Any], Any]] = {
    "add": operator.add, "sub": operator.sub, "mul": operator.mul,
    "eq": operator.eq, "ne": operator.ne, "lt": operator.lt,
    "le": operator.le, "gt": operator.gt, "ge": operator.ge,
    "and": operator.and_, "or": operator.or_,
}


def to_arrow(table: dict) -> pa.Table:
    cols = list(table["columns"].items())
    return pa.table({
        name: pa.array([row[i] for row in table["rows"]], type=ARROW_TYPES[dtype])
        for i, (name, dtype) in enumerate(cols)
    })


# ------------------------------------------------------------------ interpreter
def build_expr(e: dict):
    k = e["op"]
    if k == "col":
        return nw.col(e["name"])
    if k == "lit":
        return nw.lit(e["value"], dtype=NW_TYPES[e["dtype"]])
    if k == "bin":
        return BINARY[e["fn"]](build_expr(e["l"]), build_expr(e["r"]))
    if k == "un":
        x = build_expr(e["x"])
        return {
            "not": lambda: ~x,
            "neg": lambda: -x,
            "abs": lambda: x.abs(),
            "is_null": lambda: x.is_null(),
            "is_not_null": lambda: ~x.is_null(),
            "is_nan": lambda: x.is_nan(),
        }[e["fn"]]()
    if k == "fill_null":
        return build_expr(e["x"]).fill_null(e["value"])
    if k == "cast":
        return build_expr(e["x"]).cast(NW_TYPES[e["to"]])
    if k == "str":
        s = build_expr(e["x"]).str
        fn = e["fn"]
        if fn == "slice":
            return s.slice(e["offset"], e["length"])
        if fn == "contains":
            return s.contains(e["arg"], literal=True)
        if fn in ("starts_with", "ends_with"):
            return getattr(s, fn)(e["arg"])
        return getattr(s, fn)()
    if k == "when":
        return nw.when(build_expr(e["cond"])).then(build_expr(e["then"])).otherwise(
            build_expr(e["otherwise"]))
    if k == "over":
        return getattr(build_expr(e["x"]), e["agg"])().over(*e["by"])
    raise ValueError(f"unknown expr {k}")


def build_agg(a: dict):
    if a["fn"] == "len":
        return nw.len().alias(a["name"])
    return getattr(nw.col(a["col"]), a["fn"])().alias(a["name"])


def apply_op(frame, op: dict, right):
    k = op["op"]
    if k == "filter":
        return frame.filter(build_expr(op["pred"]))
    if k == "with_columns":
        return frame.with_columns(build_expr(op["expr"]).alias(op["name"]))
    if k == "select":
        return frame.select(op["cols"])
    if k == "group_by":
        return frame.group_by(op["keys"]).agg(*[build_agg(a) for a in op["aggs"]])
    if k == "agg":
        return frame.select(*[build_agg(a) for a in op["aggs"]])
    if k == "join":
        if op["how"] == "cross":
            return frame.join(right, how="cross")
        return frame.join(right, on=op["on"], how=op["how"])
    if k == "sort":
        return frame.sort(op["by"], descending=op["descending"], nulls_last=op["nulls_last"])
    if k == "sort_head":
        return frame.sort(op["by"], descending=op["descending"],
                          nulls_last=op["nulls_last"]).head(op["n"])
    if k == "unique":
        return frame.unique()
    if k == "drop_nulls":
        return frame.drop_nulls(subset=op["subset"])
    raise ValueError(f"unknown op {k}")


def run_pipeline(frame, pipeline: list[dict], right):
    for op in pipeline:
        frame = apply_op(frame, op, right)
    return frame


# --------------------------------------------------------------------- backends
@dataclass
class Backend:
    name: str
    load: Callable[[pa.Table, int], Any]     # (arrow table, table index) -> narwhals frame
    collect: Callable[[Any], pa.Table]      # narwhals frame -> arrow


def _polars_chunked(t: pa.Table, _idx: int):
    df = pl.from_arrow(t)
    if df.height >= 2:
        mid = df.height // 2
        df = pl.concat([df[:mid], df[mid:]], rechunk=False)
    return nw.from_native(df, eager_only=True)


def make_backends(names: list[str] | None = None) -> list[Backend]:
    con = duckdb.connect()

    def duck_load(t: pa.Table, idx: int):
        # Registering under a stable name replaces the previous case's table,
        # and makes the SQL that Narwhals generates readable for bug reports.
        con.register(f"t{idx}", t)
        return nw.from_native(con.table(f"t{idx}"))

    def polars_lazy(t: pa.Table, _idx: int):
        return nw.from_native(pl.from_arrow(t).lazy())

    def lazy_collect(**kwargs):
        return lambda f: nw.to_native(f).collect(**kwargs).to_arrow()

    all_backends = [
        Backend("polars", lambda t, _: nw.from_native(pl.from_arrow(t), eager_only=True),
                lambda f: f.to_arrow()),
        Backend("polars-chunked", _polars_chunked, lambda f: f.to_arrow()),
        Backend("polars-lazy", polars_lazy, lazy_collect()),
        Backend("polars-noopt", polars_lazy, lazy_collect(optimizations=pl.QueryOptFlags.none())),
        Backend("polars-streaming", polars_lazy, lazy_collect(engine="streaming")),
        Backend("duckdb", duck_load, lambda f: f.collect().to_arrow()),
        Backend("pyarrow", lambda t, _: nw.from_native(t, eager_only=True), lambda f: f.to_arrow()),
        # pandas results are converted with preserve_index=False rather than via
        # nw.DataFrame.to_arrow(), which leaks the index (known: narwhals-pandas-index-leak).
        Backend("pandas",
                lambda t, _: nw.from_native(t.to_pandas(types_mapper=pd.ArrowDtype), eager_only=True),
                lambda f: pa.Table.from_pandas(nw.to_native(f), preserve_index=False)),
    ]
    if names:
        unknown = set(names) - {b.name for b in all_backends}
        if unknown:
            raise ValueError(f"unknown backends: {sorted(unknown)}")
        return [b for b in all_backends if b.name in names]
    return all_backends


BACKEND_NAMES = ["polars", "polars-chunked", "polars-lazy", "polars-noopt",
                 "polars-streaming", "duckdb", "pyarrow", "pandas"]


# ---------------------------------------------------------------------- running
def classify_error(exc: BaseException) -> str:
    if isinstance(exc, ple.PanicException):
        return "panic"
    if isinstance(exc, NotImplementedError):
        return "unsupported"
    msg = str(exc).lower()
    if "not supported" in msg or "not implemented" in msg or "unsupported" in msg:
        return "unsupported"
    return "error"


def run_case(case: dict, backend: Backend, pipeline: list[dict] | None = None) -> dict:
    """Run `case` (or an alternative pipeline over its tables) on one backend."""
    pipeline = case["pipeline"] if pipeline is None else pipeline
    try:
        arrow_tables = [to_arrow(t) for t in case["tables"]]
        frames = [backend.load(t, i) for i, t in enumerate(arrow_tables)]
        right = frames[1] if len(frames) > 1 else None
        out = backend.collect(run_pipeline(frames[0], pipeline, right))
        return normalize(out, ordered=is_ordered({"pipeline": pipeline}))
    except BaseException as exc:  # PanicException derives from BaseException
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        first_line = (str(exc).strip().splitlines() or [""])[0][:300]
        return {"status": classify_error(exc), "error": f"{type(exc).__name__}: {first_line}"}
