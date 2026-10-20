"""Regression tests: each triaged upstream bug, minimised, must keep being
(1) detected by the oracles and (2) attributed to the right known issue.
If one of these starts failing after a library upgrade, the bug was probably
fixed upstream: update known.py and the report status.
"""

import pytest

from framebreaker.engines import make_backends
from framebreaker.known import classify
from framebreaker.oracle import check


def lit(v, t):
    return {"op": "lit", "value": v, "dtype": t}


def col(n):
    return {"op": "col", "name": n}


CASES = {
    "polars-scalar-column-literal-cmp": {
        "tables": [{"columns": {"a": "int"}, "rows": [[1], [2], [3]]}],
        "pipeline": [
            {"op": "with_columns", "name": "s", "expr": lit(1, "int")},
            {"op": "with_columns", "name": "x",
             "expr": {"op": "bin", "fn": "eq", "l": lit(0, "int"), "r": col("s")}},
        ],
    },
    "duckdb-arrow-pushdown-float-semantics": {
        # DuckDB turns cross join + equality filter into a hash join and pushes an IN
        # filter into the Arrow scan, where pyarrow's is_in says -0.0 != 0.0.
        "tables": [{"columns": {"c0": "float", "c1": "float"}, "rows": [[3.0, None], [-0.0, None]]},
                   {"columns": {"c1": "float", "r1": "str"}, "rows": [[0.0, ""]]}],
        "pipeline": [{"op": "join", "how": "cross", "on": []},
                     {"op": "filter", "pred": {"op": "bin", "fn": "eq", "l": col("c1_right"),
                                               "r": col("c0")}}],
    },
    "duckdb-arrow-pushdown-float-semantics/nan": {
        # TLP: NaN > 0 is true in DuckDB, but the pushed-down Arrow filter says false,
        # while NOT(NaN > 0) is evaluated by DuckDB itself - the row is in neither half.
        "tables": [{"columns": {"x": "float"}, "rows": [[float("nan")]]}],
        "pipeline": [{"op": "select", "cols": ["x"]}],
        "tlp": {"op": "bin", "fn": "gt", "l": col("x"), "r": lit(0.0, "float")},
    },
    "narwhals-pandas-literal-dtypes": {
        "tables": [{"columns": {"a": "int"}, "rows": [[1]]}],
        "pipeline": [{"op": "with_columns", "name": "b", "expr": lit(None, "bool")}],
    },
    "narwhals-duckdb-n-unique-empty": {
        "tables": [{"columns": {"a": "float"}, "rows": []}],
        "pipeline": [{"op": "agg", "aggs": [{"name": "n", "fn": "n_unique", "col": "a"}]}],
    },
    "narwhals-str-slice-negative-offset": {
        "tables": [{"columns": {"s": "str"}, "rows": [["A"]]}],
        "pipeline": [{"op": "with_columns", "name": "t",
                      "expr": {"op": "str", "fn": "slice", "x": col("s"), "offset": -2, "length": 2}}],
    },
    "narwhals-pyarrow-empty-suffix": {
        "tables": [{"columns": {"s": "str"}, "rows": [["ab"]]}],
        "pipeline": [{"op": "filter", "pred": {"op": "str", "fn": "ends_with", "x": col("s"), "arg": ""}}],
    },
    "polars-median-neg-inf": {
        "tables": [{"columns": {"k": "int", "x": "float"}, "rows": [[1, 0.0], [1, float("-inf")]]}],
        "pipeline": [{"op": "group_by", "keys": ["k"], "aggs": [{"name": "m", "fn": "median", "col": "x"}]}],
    },
    "pandas-groupby-nan-sentinel": {
        "tables": [{"columns": {"k": "str", "x": "float"}, "rows": [["a", float("nan")], ["b", 1.0]]}],
        "pipeline": [{"op": "group_by", "keys": ["k"], "aggs": [{"name": "m", "fn": "min", "col": "x"}]}],
    },
    "pandas-negzero-lexsort": {
        "tables": [{"columns": {"f": "float", "s": "str"}, "rows": [[-0.0, ""], [0.0, ""]]}],
        "pipeline": [{"op": "sort", "by": ["f", "s"], "descending": [False, False], "nulls_last": False}],
    },
}


@pytest.fixture(scope="module")
def backends():
    return make_backends()


@pytest.mark.parametrize("known_id", sorted(CASES))
def test_known_bug_still_reproduces_and_is_classified(known_id, backends):
    case = CASES[known_id]
    verdict = check(case, backends)
    assert verdict is not None, f"{known_id} no longer reproduces - fixed upstream?"
    label = classify(verdict, case, rerun=lambda c: check(c, backends))
    assert label is not None and known_id.split("/")[0] in label.split("+"), label


# ------------------------------------------------- triage must not hide new bugs
def test_incidental_nan_does_not_hide_a_real_bug():
    """A planted bug (ints with nulls read as 0) in a case that *also* contains NaN.
    The NaN pattern matches, but removing the NaN does not fix the disagreement,
    so counterfactual triage must report the finding as new."""
    import narwhals as nw
    import polars as pl

    from framebreaker.engines import Backend
    buggy = Backend("pandas",  # impersonate a real backend so the pandas rules apply
                    lambda t, _: nw.from_native(pl.from_arrow(t), eager_only=True),
                    lambda f: nw.to_native(f).with_columns(pl.col(pl.Int64).fill_null(0)).to_arrow())
    backends = [b for b in make_backends() if b.name != "pandas"] + [buggy]
    case = {"tables": [{"columns": {"i": "int", "f": "float"}, "rows": [[None, float("nan")], [2, 1.0]]}],
            "pipeline": [{"op": "select", "cols": ["i", "f"]}]}
    verdict = check(case, backends)
    assert verdict is not None
    assert classify(verdict, case) is not None                      # pattern-only triage hides it...
    assert classify(verdict, case, rerun=lambda c: check(c, backends)) is None  # ...counterfactual does not


def test_two_causes_in_one_backend_are_peeled():
    """pyarrow is wrong twice here: -0.0 hashing (n_unique) and its approximate median."""
    case = {"tables": [{"columns": {"c0": "int", "c1": "float", "c2": "str"},
                        "rows": [[0, -0.0, None], [9999, 0.0, None]]}],
            "pipeline": [{"op": "group_by", "keys": ["c2"],
                          "aggs": [{"name": "a0", "fn": "n_unique", "col": "c1"},
                                   {"name": "a1", "fn": "median", "col": "c0"}]}]}
    backends = make_backends()
    verdict = check(case, backends)
    label = classify(verdict, case, rerun=lambda c: check(c, backends))
    assert label is not None and "arrow-negzero-hash" in label
