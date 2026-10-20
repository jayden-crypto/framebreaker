import json
import math
import random

import narwhals as nw
import polars as pl
import pytest

from framebreaker.compare import canon, normalize, values_equal
from framebreaker.engines import Backend, make_backends, run_case
from framebreaker.gen import Generator
from framebreaker.ir import InvalidCase, data_tags, features, validate
from framebreaker.oracle import check, fingerprint, tlp_valid
from framebreaker.render import polars_script
from framebreaker.shrink import shrink


def case_of(pipeline, columns=None, rows=None, right=None):
    tables = [{"columns": columns or {"a": "int", "s": "str"},
               "rows": rows if rows is not None else [[1, "x"], [None, "y"], [3, None]]}]
    if right:
        tables.append(right)
    return {"tables": tables, "pipeline": pipeline}


COL_A = {"op": "col", "name": "a"}


def lit(v, t):
    return {"op": "lit", "value": v, "dtype": t}


# ------------------------------------------------------------------------ IR
def test_validate_tracks_schema_through_ops():
    case = case_of([
        {"op": "with_columns", "name": "b", "expr": {"op": "bin", "fn": "add", "l": COL_A, "r": lit(1.5, "float")}},
        {"op": "group_by", "keys": ["s"], "aggs": [{"name": "m", "fn": "mean", "col": "b"}]},
    ])
    assert validate(case) == {"s": "str", "m": "float"}


@pytest.mark.parametrize("pipeline", [
    [{"op": "filter", "pred": COL_A}],                                   # non-bool predicate
    [{"op": "select", "cols": ["nope"]}],                                # unknown column
    [{"op": "with_columns", "name": "x", "expr": {"op": "bin", "fn": "add", "l": COL_A,
                                                    "r": lit("a", "str")}}],  # int + str
    [{"op": "sort_head", "by": ["a"], "descending": [False], "nulls_last": True, "n": 1}],
])
def test_validate_rejects_ill_typed_cases(pipeline):
    with pytest.raises(InvalidCase):
        validate(case_of(pipeline))


def test_tags_and_features_cover_literals():
    case = case_of([{"op": "filter", "pred": {"op": "bin", "fn": "gt", "l": lit(-0.0, "float"),
                                              "r": COL_A}}])
    assert "neg-zero" in data_tags(case)
    assert "bin:gt" in features(case) and "filter" in features(case)


# ------------------------------------------------------------------- generator
def test_generator_is_deterministic_and_valid():
    gen1, gen2 = Generator(random.Random(7)), Generator(random.Random(7))
    for _ in range(200):
        c1, c2 = gen1.gen_case(), gen2.gen_case()
        assert json.dumps(c1) == json.dumps(c2)
        validate(c1)
        validate(json.loads(json.dumps(c1)))  # JSON round-trips (incl. NaN) to a valid case


def test_backend_names_constant_matches_registry():
    from framebreaker.engines import BACKEND_NAMES
    assert BACKEND_NAMES == [b.name for b in make_backends()]


# --------------------------------------------------------------------- compare
def test_canon_equates_signed_zero_and_nan():
    assert canon(-0.0) == canon(0.0)
    assert canon(float("nan")) == canon(float("nan"))
    assert canon(None) != canon(0)
    assert canon(1) == canon(1.0)  # values equal; dtypes are compared separately


def test_normalize_is_order_insensitive_unless_ordered():
    import pyarrow as pa
    t1 = pa.table({"x": [3, 1, 2]})
    t2 = pa.table({"x": [1, 2, 3]})
    assert values_equal(normalize(t1, ordered=False), normalize(t2, ordered=False))
    assert not values_equal(normalize(t1, ordered=True), normalize(t2, ordered=True))


def test_float_tolerance():
    import pyarrow as pa
    a = normalize(pa.table({"x": [0.1 + 0.2]}), ordered=False)
    b = normalize(pa.table({"x": [0.3]}), ordered=False)
    assert values_equal(a, b)
    c = normalize(pa.table({"x": [math.inf]}), ordered=False)
    assert not values_equal(a, c)


# ---------------------------------------------------------------------- engines
def test_all_backends_agree_on_a_plain_pipeline():
    case = case_of([
        {"op": "filter", "pred": {"op": "un", "fn": "is_not_null", "x": COL_A}},
        {"op": "with_columns", "name": "b", "expr": {"op": "bin", "fn": "mul", "l": COL_A, "r": lit(2, "int")}},
    ])
    assert check(case, make_backends()) is None
    out = run_case(case, make_backends(["duckdb"])[0])
    assert out["columns"] == ["a", "s", "b"] and len(out["rows"]) == 2


# ----------------------------------------------------- planted bugs (end to end)
def _polars_with(transform):
    """A Polars backend with a deliberately injected bug."""
    return Backend("buggy",
                   lambda t, _: nw.from_native(pl.from_arrow(t), eager_only=True),
                   lambda f: transform(nw.to_native(f)).to_arrow())


def _null_ints_become_zero(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.col(pl.Int64).fill_null(0))


def test_differential_oracle_finds_and_shrinks_planted_bug():
    backends = make_backends(["polars", "duckdb"]) + [_polars_with(_null_ints_become_zero)]
    gen = Generator(random.Random(0))
    for _ in range(500):
        case = gen.gen_case()
        verdict = check(case, backends)
        if verdict and "buggy" in sum(verdict.get("partition", []), []):
            break
    else:
        pytest.fail("planted bug not found in 500 cases")
    fp = fingerprint(verdict)
    small, _ = shrink(case, lambda c: (v := check(c, backends)) is not None and fingerprint(v) == fp,
                      tlp_valid)
    assert len(small["pipeline"]) <= 2
    assert sum(len(t["rows"]) for t in small["tables"]) <= 2
    assert "null" in data_tags(small)


def test_tlp_oracle_catches_bug_without_a_reference_engine():
    # A "filter" that keeps rows where the predicate is null: TLP must notice the
    # duplicated rows even though no other backend is consulted.
    class Sneaky:
        name = "polars-lazy"   # TLP runs on this backend name

        def __init__(self):
            self.inner = make_backends(["polars"])[0]

        def load(self, t, i):
            return self.inner.load(t, i)

        def collect(self, f):
            return self.inner.collect(f)

    import framebreaker.engines as engines
    original = engines.build_expr

    def leaky(e):
        if e.get("op") == "un" and e.get("fn") == "not":
            return ~original(e["x"]).fill_null(False)  # NOT(null) wrongly becomes true
        return original(e)

    engines.build_expr = leaky
    try:
        case = case_of([{"op": "select", "cols": ["a"]}])
        case["tlp"] = {"op": "bin", "fn": "gt", "l": COL_A, "r": lit(1, "int")}
        verdict = check(case, [Sneaky()])
    finally:
        engines.build_expr = original
    assert verdict is not None and verdict["kind"] == "tlp"


# ---------------------------------------------------------------------- render
def test_polars_reproducer_runs(capsys):
    case = case_of([
        {"op": "filter", "pred": {"op": "bin", "fn": "ge", "l": COL_A, "r": lit(1, "int")}},
        {"op": "group_by", "keys": ["s"], "aggs": [{"name": "n", "fn": "len", "col": None}]},
    ])
    for mode in ("eager", "lazy", "noopt", "streaming", "chunked"):
        exec(compile(polars_script(case, mode), "<repro>", "exec"), {})
    assert "shape" in capsys.readouterr().out
