"""Registry of already-understood discrepancies, with counterfactual triage.

Each entry recognises a family of findings: either a confirmed bug (upstream
link, or a draft report under reports/ if not yet filed) or an intentional
semantic difference between libraries. Matching findings are still saved, but
under `known/`, so new bugs are not drowned out by ones already triaged. The
registry doubles as an atlas of where dataframe libraries disagree.

Pattern matching alone is dangerous: "the data contains NaN" does not mean
NaN *caused* the disagreement, and a broad rule silently hides new bugs. So
where a cause can be removed without changing what the query means, an entry
carries a `remove_cause` transform. The finding only counts as explained if the
disagreement disappears once the cause is removed (NaN replaced by an ordinary
number, -0.0 by 0.0, a literal moved to the other side of a comparison...).
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Callable, Optional

from .ir import data_tags, features, iter_exprs

POLARS_FAMILY = {"polars", "polars-chunked", "polars-lazy", "polars-noopt", "polars-streaming"}
Rerun = Callable[[dict], Optional[dict]]


# ------------------------------------------------------------------- helpers
def _errors(verdict: dict) -> str:
    return " ".join(o.get("error", "") for o in verdict.get("outcomes", {}).values())


def _minority(verdict: dict) -> set[str]:
    """The backends that disagree with the largest group (or that failed alone)."""
    if verdict.get("minority"):
        return set(verdict["minority"])
    if verdict.get("partition"):
        groups = sorted(verdict["partition"], key=len)
        return set().union(*groups[:-1])
    return set(verdict.get("backends") or [])


def _only(verdict: dict, *names: str) -> bool:
    m = _minority(verdict)
    return bool(m) and m <= set(names)


def _exprs(case: dict, op: str, **attrs) -> list[dict]:
    return [e for e in iter_exprs([case["pipeline"], case.get("tlp")])
            if e["op"] == op and all(e.get(k) == v for k, v in attrs.items())]


def _polars_disagree(verdict: dict | None) -> bool:
    if verdict is None:
        return False
    groups = verdict.get("partition") or []
    return sum(1 for g in groups if set(g) & POLARS_FAMILY) > 1


def _outcome(verdict: dict, names) -> dict | None:
    for n in names:
        o = verdict.get("outcomes", {}).get(n)
        if o and o.get("status") == "ok":
            return o
    return None


def _reference(verdict: dict) -> list[str]:
    groups = verdict.get("partition") or []
    minority = _minority(verdict)
    rest = [g for g in groups if not set(g) & minority]
    return max(rest, key=len) if rest else []


def _null_column_where_reference_has(verdict: dict, values: set) -> bool:
    """A minority backend returns null where the reference returns one of `values`."""
    ref = _outcome(verdict, _reference(verdict))
    if ref is None:
        return False
    for name in _minority(verdict):
        o = _outcome(verdict, [name])
        if o is None or o["columns"] != ref["columns"] or len(o["rows"]) != len(ref["rows"]):
            continue
        for ro, rr in zip(o["rows"], ref["rows"]):
            for vo, vr in zip(ro, rr):
                if tuple(vo)[0] == 0 and tuple(vr)[0] != 0 and tuple(vr)[1] in values:
                    return True
    return False


def _differing_columns(verdict: dict) -> set[str]:
    ref = _outcome(verdict, _reference(verdict))
    diff: set[str] = set()
    if ref is None:
        return diff
    for name in _minority(verdict):
        o = _outcome(verdict, [name])
        if o is None or o["columns"] != ref["columns"] or len(o["rows"]) != len(ref["rows"]):
            return set(ref["columns"])  # shape differs: everything is suspect
        for ro, rr in zip(o["rows"], ref["rows"]):
            for col, vo, vr in zip(ref["columns"], ro, rr):
                if tuple(vo) != tuple(vr):
                    diff.add(col)
    return diff


def _reference_has_null_values(verdict: dict) -> bool:
    ref = _outcome(verdict, _reference(verdict))
    return ref is not None and any(tuple(v)[0] == 0 for row in ref["rows"] for v in row)


def _minority_has_null_dtype(verdict: dict) -> bool:
    return any("null" in (verdict["outcomes"].get(n) or {}).get("dtypes", []) for n in _minority(verdict))


# --------------------------------------------------- counterfactual transforms
def _map_values(case: dict, fn) -> dict:
    """Apply fn to every data value and literal in a copy of the case."""
    new = copy.deepcopy(case)
    for t in new["tables"]:
        t["rows"] = [[fn(v) for v in row] for row in t["rows"]]
    for e in iter_exprs([new["pipeline"], new.get("tlp")]):
        if e["op"] in ("lit", "fill_null"):
            e["value"] = fn(e["value"])
    return new


def _finite(case: dict) -> dict:
    """NaN and +/-inf replaced by ordinary numbers."""
    def fix(v):
        if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
            return 0.25 if v != v else (1e6 if v > 0 else -1e6)
        return v
    return _map_values(case, fix)


def _no_negzero(case: dict) -> dict:
    """-0.0 replaced by 0.0 in the data, and float columns computed by the pipeline wrapped
    in `+ 0.0` (IEEE: -0.0 + 0.0 == +0.0), so arithmetic cannot produce -0.0 either."""
    from .ir import InvalidCase, apply_schema, expr_dtype, table_schema
    new = _map_values(case, lambda v: 0.0 if isinstance(v, float) and v == 0 else v)
    schema = table_schema(new["tables"][0])
    right = table_schema(new["tables"][1]) if len(new["tables"]) > 1 else None
    try:
        for op in new["pipeline"]:
            if op["op"] == "with_columns" and expr_dtype(op["expr"], schema) == "float":
                op["expr"] = {"op": "bin", "fn": "add", "l": op["expr"],
                              "r": {"op": "lit", "value": 0.0, "dtype": "float"}}
            schema = apply_schema(op, schema, right)
    except InvalidCase:
        return new
    return new


def _ascii_strings(case: dict) -> dict:
    return _map_values(case, lambda v: v.encode("ascii", "replace").decode() if isinstance(v, str) else v)


MIRROR = {"eq": "eq", "ne": "ne", "lt": "gt", "gt": "lt", "le": "ge", "ge": "le"}


def _literal_on_right(case: dict) -> dict:
    new = copy.deepcopy(case)
    for e in iter_exprs([new["pipeline"], new.get("tlp")]):
        if e["op"] == "bin" and e["fn"] in MIRROR and e["l"]["op"] == "lit":
            e["l"], e["r"], e["fn"] = e["r"], e["l"], MIRROR[e["fn"]]
    return new


def _materialize_literals(case: dict) -> dict:
    """Replace literals by real input columns holding the same value.

    Only applicable when every literal sits in a first, row-wise op (filter or
    with_columns), where input columns are in scope; otherwise the case is
    returned unchanged, meaning "cause cannot be removed, trust the pattern".
    """
    first = case["pipeline"][0] if case["pipeline"] else None
    later = [e for e in iter_exprs([case["pipeline"][1:], case.get("tlp")]) if e["op"] == "lit"]
    if first is None or first["op"] not in ("filter", "with_columns") or later:
        return case
    new = copy.deepcopy(case)
    table = new["tables"][0]
    n = 0
    for e in iter_exprs(new["pipeline"][0]):
        if e["op"] == "lit":
            name = f"_lit{n}"
            n += 1
            table["columns"][name] = e["dtype"]
            for row in table["rows"]:
                row.append(e["value"])
            e.clear()
            e.update({"op": "col", "name": name})
    return new


# ----------------------------------------------------------------- registry
@dataclass(frozen=True)
class Known:
    id: str
    kind: str          # "bug" or "semantics"
    title: str
    link: str          # upstream issue, or the draft report in reports/
    match: Callable[[dict, dict], bool]
    remove_cause: Callable[[dict], dict] | None = None
    # stricter pattern, used when remove_cause is not applicable to a case
    fallback: Callable[[dict, dict], bool] | None = None


def _both(case: dict) -> dict:
    return _no_negzero(_finite(case))


# Polars-internal bugs: matched even though Polars variants disagree with each other.
POLARS_KNOWN: list[Known] = [
    Known(
        "polars-scalar-column-literal-cmp", "bug",
        "Polars: `lit(x) <cmp> col(s)` returns 1 row when column s is backed by a scalar "
        "(in-memory engine only; regression in 1.14.0)",
        "reports/polars-scalar-column-literal-cmp.md",
        lambda v, c: any(e["fn"] in MIRROR and e["l"]["op"] == "lit" for e in _exprs(c, "bin"))
        and _polars_disagree(v),
        remove_cause=_literal_on_right,
    ),
]

KNOWN: list[Known] = [
    # ---------------------------------------------------------------- bugs
    Known(
        "pandas-merge-bool-mixed-backend", "bug",
        "pandas: merge on numpy bool vs bool[pyarrow] containing NA raises ArrowInvalid",
        "reports/pandas-merge-bool-mixed-backend.md",
        lambda v, c: "tried to convert to boolean" in _errors(v) and _only(v, "pandas"),
    ),
    Known(
        "duckdb-arrow-pushdown-float-semantics", "bug",
        "DuckDB: filters pushed into Arrow scans use Arrow's float semantics, not DuckDB's: "
        "`x > 0` drops NaN rows and IN/join filters treat -0.0 != 0.0, so rows silently disappear",
        "reports/duckdb-arrow-pushdown-float-semantics.md",
        lambda v, c: _only(v, "duckdb") and (
            "neg-zero" in data_tags(c) or ("nan" in data_tags(c) and v["kind"] == "tlp")),
        remove_cause=_both,
    ),
    Known(
        "duckdb-python-nan-constant", "bug",
        "DuckDB Python: ConstantExpression(float('nan')) becomes NULL (bind parameters were fixed in "
        "duckdb#4623; this path was not)",
        "reports/duckdb-python-nan-constant.md",
        lambda v, c: any(isinstance(e["value"], float) and e["value"] != e["value"]
                         for e in _exprs(c, "lit")) and "duckdb" in _minority(v),
        remove_cause=_finite,
    ),
    Known(
        "narwhals-pandas-index-leak", "bug",
        "narwhals: DataFrame.to_arrow() on pandas leaks the index as __index_level_0__ "
        "once a filter/sort leaves a non-contiguous index (worked around in engines.py)",
        "reports/narwhals-pandas-to-arrow-index.md",
        lambda v, c: any("__index_level_0__" in o.get("columns", [])
                         for o in v.get("outcomes", {}).values()),
    ),
    Known(
        "narwhals-pandas-literal-dtypes", "bug",
        "narwhals: pandas-like lit() ignores the frame's dtype backend: lit(None, Boolean) "
        "becomes False, lit(None, Int64) raises, string literals get NaN-semantics `str` dtype",
        "reports/narwhals-pandas-literal-dtypes.md",
        lambda v, c: _only(v, "pandas") and bool(_exprs(c, "lit")),
        remove_cause=_materialize_literals,
        fallback=lambda v, c: any(e["value"] is None for e in _exprs(c, "lit"))
        or any(e["dtype"] == "str" for e in _exprs(c, "lit"))
        or "int() argument must be" in _errors(v),
    ),
    Known(
        "narwhals-duckdb-n-unique-empty", "bug",
        "narwhals: n_unique() on an empty DuckDB relation returns NULL instead of 0",
        "reports/narwhals-duckdb-n-unique-empty.md",
        lambda v, c: _only(v, "duckdb") and "agg:n_unique" in features(c),
    ),
    Known(
        "narwhals-str-slice-negative-offset", "bug",
        "narwhals: str.slice with a negative offset is wrong on DuckDB when |offset| > len, "
        "and on pyarrow/pandas when offset + length >= 0",
        "reports/narwhals-str-slice-negative-offset.md",
        lambda v, c: any(e.get("offset", 0) < 0 for e in _exprs(c, "str", fn="slice"))
        and _only(v, "duckdb", "pyarrow", "pandas"),
    ),
    Known(
        "narwhals-pyarrow-empty-suffix", "bug",
        "narwhals: str.ends_with('') returns False on pyarrow for non-empty strings",
        "reports/narwhals-pyarrow-ends-with-empty.md",
        lambda v, c: any(e.get("arg") == "" for e in _exprs(c, "str", fn="ends_with"))
        and _only(v, "pyarrow"),
    ),
    Known(
        "narwhals-str-pattern-literal", "bug",
        "narwhals: str.contains/starts_with/ends_with on a literal receiver raise "
        "TypeError on pyarrow and pandas",
        "reports/narwhals-str-method-literal-receiver.md",
        lambda v, c: ("only supports" in _errors(v) or "has no attribute 'as_py'" in _errors(v))
        and _only(v, "pyarrow", "pandas"),
    ),
    Known(
        "narwhals-agg-null-dtype", "bug",
        "narwhals: an aggregation whose result is null comes back null-typed on pyarrow "
        "(and object/float on pandas) instead of keeping its dtype",
        "reports/narwhals-agg-null-dtype.md",
        lambda v, c: _only(v, "pyarrow", "pandas") and any(f.startswith("agg:") for f in features(c)) and (
            _minority_has_null_dtype(v)
            or (v["kind"] == "dtype" and ("empty-table" in data_tags(c) or _reference_has_null_values(v)))
            or any(m in _errors(v) for m in ("Invalid null value", "NAType", "(null,",
                                              "Data type null is not supported",
                                              "only supported for numeric dtype and not String"))),
    ),
    Known(
        "pandas-negzero-lexsort", "bug",
        "pandas: multi-column sort_values raises 'Categorical categories must be unique' "
        "on pyarrow floats containing both -0.0 and 0.0",
        "https://github.com/pandas-dev/pandas/issues/54911",
        lambda v, c: any(m in _errors(v) for m in ("Categorical categories must be unique",
                                                   "values should be unique if codes")),
    ),
    Known(
        "polars-median-neg-inf", "bug",
        "Polars: median/quantile(linear) returns NaN when the lower middle value is -inf "
        "(lo + (hi - lo) * t gives -inf + inf); NumPy/DuckDB/pandas return -inf",
        "reports/polars-median-neg-inf.md",
        lambda v, c: _only(v, *POLARS_FAMILY) and "inf" in data_tags(c)
        and any("median" in f for f in features(c)),
        remove_cause=_finite,
    ),
    Known(
        "pandas-groupby-nan-sentinel", "bug",
        "pandas: groupby min/max of an all-NaN group returns +inf/-inf (internal sentinels leak); "
        "reported for masked Float dtypes, also affects Arrow-backed floats",
        "https://github.com/pandas-dev/pandas/issues/59106",
        lambda v, c: _only(v, "pandas") and "nan" in data_tags(c)
        and {"agg:min", "agg:max"} & set(features(c)) and "group_by" in features(c),
        remove_cause=_finite,
    ),
    Known(
        "arrow-negzero-hash", "bug",
        "Arrow hashes -0.0 and 0.0 as different values (unique/group_by/is_in)",
        "https://github.com/apache/arrow/issues/26214",
        lambda v, c: "neg-zero" in data_tags(c) and _only(v, "pyarrow", "pandas"),
        remove_cause=_no_negzero,
    ),
    # ----------------------------------------------------------- semantics
    Known(
        "duckdb-variance-nonfinite", "semantics",
        "DuckDB raises 'out of range' for std/var over NaN or inf instead of returning NaN",
        "https://github.com/duckdb/duckdb/issues/14373",
        lambda v, c: "is out of range" in _errors(v) and _only(v, "duckdb"),
        remove_cause=_finite,
    ),
    Known(
        "pyarrow-approximate-median", "semantics",
        "pyarrow computes grouped median approximately (t-digest); Narwhals documents "
        "that median may differ across backends",
        "https://narwhals-dev.github.io/narwhals/api-reference/expr/#narwhals.Expr.median",
        lambda v, c: _only(v, "pyarrow") and bool(_differing_columns(v)) and _differing_columns(v) <= {
            a["name"] for e in c["pipeline"] for a in e.get("aggs", []) if a["fn"] == "median"},
    ),
    Known(
        "unicode-case-mapping", "semantics",
        "Case mapping differs: Polars uses full Unicode mappings (ß -> SS), "
        "utf8proc-based engines (Arrow, DuckDB) map ß -> ẞ",
        "",
        lambda v, c: "unicode" in data_tags(c) and any(
            f in features(c) for f in ("str:to_uppercase", "str:to_lowercase")),
        remove_cause=_ascii_strings,
    ),
    Known(
        "integer-width", "semantics",
        "len()/count()/len_chars() return different integer widths per backend (Polars u32, "
        "Arrow int32, others int64): unsigned arithmetic wraps, negation and joins differ",
        "",
        lambda v, c: "Incompatible data types for corresponding join field keys" in _errors(v)
        or "not supported for dtype `u32`" in _errors(v)
        or (_only(v, *POLARS_FAMILY)
            and {"agg:len", "agg:count", "str:len_chars"} & set(features(c))
            and {"bin:sub", "bin:add", "bin:mul", "un:neg"} & set(features(c))),
    ),
    Known(
        "pandas-nan-as-missing", "semantics",
        "pandas (numpy-backed results) stores missing floats as NaN, so is_nan() is true for "
        "values that are null in other backends (e.g. std of a single value)",
        "",
        lambda v, c: _only(v, "pandas") and (
            "un:is_nan" in features(c)
            or ({"agg:std", "agg:mean", "agg:median"} & set(features(c)) and _reference_has_null_values(v))),
    ),
    Known(
        "nan-semantics", "semantics",
        "NaN handling differs by design: Polars/DuckDB order NaN above all numbers and "
        "treat NaN == NaN; pandas treats NaN as missing; aggregations differ on NaN",
        "",
        lambda v, c: ("nan" in data_tags(c) or "inf" in data_tags(c))
        and v["kind"] in ("mismatch", "dtype", "error"),
        remove_cause=_finite,
    ),
]


# ----------------------------------------------------------- classification
def _never_known(verdict: dict) -> bool:
    """Polars variants disagreeing among themselves, or a Polars panic/TLP failure,
    can never be a by-design difference between libraries."""
    if verdict["kind"] in ("crash", "hang", "flaky"):
        return True
    if verdict["kind"] in ("panic", "tlp"):
        return bool(set(verdict.get("backends") or []) & POLARS_FAMILY)
    return _polars_disagree(verdict)


def _still_disagrees(after: dict | None, sub: dict, polars_internal: bool) -> bool:
    """Does the disagreement described by `sub` survive in the counterfactual verdict?"""
    if after is None:
        return False
    if polars_internal:
        return _polars_disagree(after)
    if sub["kind"] in ("tlp", "panic"):
        return after["kind"] == sub["kind"] and bool(set(after.get("backends") or []) & set(sub["backends"]))
    groups = after.get("partition") or []
    where = {n: i for i, g in enumerate(groups) for n in g}
    reference = {where[n] for n in sub["partition"][0] if n in where}
    return any(n in where and where[n] not in reference for n in _minority(sub))


class _Cache:
    def __init__(self, rerun: Rerun | None):
        self.rerun, self.memo = rerun, {}

    def __call__(self, case: dict) -> dict | None:
        key = json.dumps(case, sort_keys=True, default=str)
        if key not in self.memo:
            self.memo[key] = self.rerun(case)
        return self.memo[key]


def _confirmed(k: Known, sub: dict, case: dict, rerun: _Cache | None, polars_internal: bool) -> bool:
    try:
        if not k.match(sub, case):
            return False
        if k.remove_cause is None or rerun is None:
            return k.fallback is None or rerun is not None or k.fallback(sub, case)
        counterfactual = k.remove_cause(case)
        if counterfactual is case:   # cause cannot be removed for this case
            return k.fallback is None or k.fallback(sub, case)
        return not _still_disagrees(rerun(counterfactual), sub, polars_internal)
    except Exception:
        return False


def _peel(sub: dict, case: dict, rerun: _Cache, depth: int):
    """One backend can be wrong for several reasons at once (e.g. -0.0 hashing *and* an
    approximate median). For each removable cause whose pattern matches, remove it, re-run,
    and try to explain whatever disagreement is left."""
    if depth <= 0:
        return None
    for k in KNOWN:
        if k.remove_cause is None:
            continue
        try:
            if not k.match(sub, case):
                continue
            new_case = k.remove_cause(case)
            if new_case is case:
                continue
            after = rerun(new_case)
            if not _still_disagrees(after, sub, False):
                return (k,)
            rest = classify(after, new_case, rerun, _depth=depth - 1) if after else None
            if rest:
                return (k, *[x for x in KNOWN + POLARS_KNOWN if x.id in rest.split("+")])
        except Exception:
            continue
    return None


def _classify_one(sub: dict, case: dict, rerun: _Cache | None, depth: int = 2) -> Known | tuple | None:
    for k in POLARS_KNOWN:
        if _confirmed(k, sub, case, rerun, polars_internal=True):
            return k
    if _never_known(sub):
        return None
    for k in KNOWN:
        if _confirmed(k, sub, case, rerun, polars_internal=False):
            return k
    if rerun is not None:
        return _peel(sub, case, rerun, depth)
    return None


def _ids(k) -> list[str]:
    return [x.id for x in k] if isinstance(k, tuple) else [k.id]


def classify(verdict: dict, case: dict, rerun: Rerun | None = None, _depth: int = 2) -> str | None:
    """Return the id(s) of known issues explaining this finding, or None if it is new.

    `rerun(case) -> verdict | None` re-checks a modified case; with it, causes are
    confirmed counterfactually, without it only the patterns are checked.
    When several groups disagree with the majority, every group must be explained.
    """
    cache = rerun if isinstance(rerun, _Cache) else (_Cache(rerun) if rerun else None)
    groups = verdict.get("partition") or []
    if len(groups) <= 1:
        k = _classify_one(verdict, case, cache, _depth)
        return "+".join(_ids(k)) if k else None
    # Polars is the reference semantics for Narwhals, so a group containing it is
    # tried as the reference first; but Polars itself can be the odd one out.
    polars_groups = [g for g in groups if set(g) & POLARS_FAMILY]
    others = sorted((g for g in groups if g not in polars_groups), key=len, reverse=True)
    if len(polars_groups) > 1:   # Polars variants disagree: try Polars-internal explanations first
        k = _classify_one(verdict, case, cache, _depth)
        if k is not None:
            return "+".join(_ids(k))
    # A counterfactual proves what *causes* a disagreement, not which side is wrong, so
    # several references can "explain" a finding. Prefer the explanation that leans least
    # on generic by-design semantics (ties: Polars-as-reference first, then fewer causes).
    generic = {k.id for k in KNOWN if k.kind == "semantics"}
    best = None
    for rank, reference in enumerate(polars_groups + others):
        ids = _explain_against(verdict, case, groups, reference, cache, _depth)
        if ids is None:
            continue
        ids = sorted(set(ids))
        score = (sum(i in generic for i in ids), rank, len(ids))
        if best is None or score < best[0]:
            best = (score, ids)
    return "+".join(best[1]) if best else None


def _explain_against(verdict: dict, case: dict, groups: list[list[str]], reference: list[str],
                     cache: _Cache | None, depth: int) -> list[str] | None:
    def explain(members: list[str]):
        sub = {**verdict, "partition": [reference, members], "minority": members,
               "outcomes": {n: o for n, o in verdict["outcomes"].items() if n in reference + members}}
        return _classify_one(sub, case, cache, depth)

    ids = []
    for g in groups:
        if g is reference:
            continue
        k = explain(g)
        if k is not None:
            ids += _ids(k)
            continue
        # Two backends can be wrong in the same way for different reasons.
        singles = [explain([name]) for name in g] if len(g) > 1 else [None]
        if any(x is None for x in singles):
            return None
        ids += [i for x in singles for i in _ids(x)]
    return ids


def signature(verdict: dict, case: dict) -> str:
    """Stable identity of a finding, used for de-duplication across runs."""
    from .oracle import fingerprint
    kind, who = fingerprint(verdict)
    return f"{kind}|{who}|{','.join(features(case))}"
