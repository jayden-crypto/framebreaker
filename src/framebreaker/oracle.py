"""Bug oracles: deciding, without a reference implementation, that something is wrong.

1. Crash oracle      - a Rust panic is always a bug, whatever other engines say.
2. Differential      - all engines must agree on the result (or all fail).
3. Partition (TLP)   - for any predicate p, the rows where p is true, false and
                       null must add up to the whole result. Ported from
                       SQLancer's Ternary Logic Partitioning. It checks a single
                       engine against itself, so it is immune to legitimate
                       semantic differences between libraries (NaN ordering etc.).
"""

from __future__ import annotations

from .compare import multiset_union, sort_key, values_equal
from .engines import Backend, run_case
from .ir import InvalidCase, apply_schema, validate

TLP_BACKENDS = ("polars-lazy", "duckdb", "pyarrow")


def _partition(outcomes: dict[str, dict], key) -> list[list[str]]:
    """Group backend names into classes of equal outcomes."""
    classes: list[tuple[dict, list[str]]] = []
    for name, out in outcomes.items():
        for rep, members in classes:
            if key(rep, out):
                members.append(name)
                break
        else:
            classes.append((out, [name]))
    return sorted(sorted(m) for _, m in classes)


def _same_values(a: dict, b: dict) -> bool:
    if a["status"] != "ok" or b["status"] != "ok":
        return a["status"] == b["status"] == "error"
    return values_equal(a, b)


def _same_dtypes(a: dict, b: dict) -> bool:
    return a["dtypes"] == b["dtypes"]


def tlp_pipelines(case: dict) -> list[list[dict]] | None:
    pred = case.get("tlp")
    if pred is None:
        return None
    base = case["pipeline"]
    return [
        base + [{"op": "filter", "pred": pred}],
        base + [{"op": "filter", "pred": {"op": "un", "fn": "not", "x": pred}}],
        base + [{"op": "filter", "pred": {"op": "un", "fn": "is_null", "x": pred}}],
    ]


def check(case: dict, backends: list[Backend], hook=None) -> dict | None:
    """Return a verdict dict describing the first failure, or None if all oracles pass.

    `hook(case)` is called before running anything; the campaign uses it to
    record the in-flight case so a segfault can still be attributed.
    """
    if hook:
        hook(case)
    outcomes = {b.name: run_case(case, b) for b in backends}

    panics = sorted(n for n, o in outcomes.items() if o["status"] == "panic")
    if panics:
        return {"kind": "panic", "backends": panics,
                "error": outcomes[panics[0]]["error"], "outcomes": outcomes}

    usable = {n: o for n, o in outcomes.items() if o["status"] != "unsupported"}
    if len(usable) >= 2:
        groups = _partition(usable, _same_values)
        if len(groups) > 1:
            statuses = {o["status"] for o in usable.values()}
            kind = "error" if "error" in statuses else "mismatch"
            return {"kind": kind, "partition": groups, "outcomes": outcomes}
        ok = {n: o for n, o in usable.items() if o["status"] == "ok"}
        if len(ok) >= 2:
            dgroups = _partition(ok, _same_dtypes)
            if len(dgroups) > 1:
                return {"kind": "dtype", "partition": dgroups, "outcomes": outcomes}

    pipelines = tlp_pipelines(case)
    if pipelines:
        for b in backends:
            if b.name not in TLP_BACKENDS or outcomes[b.name]["status"] != "ok":
                continue
            parts = [run_case(case, b, p) for p in pipelines]
            if any(p["status"] == "panic" for p in parts):
                err = next(p["error"] for p in parts if p["status"] == "panic")
                return {"kind": "panic", "backends": [b.name], "error": err, "outcomes": outcomes}
            if any(p["status"] != "ok" for p in parts):
                continue
            union = multiset_union(parts)
            base = dict(outcomes[b.name])
            base_unordered = {**base, "rows": sorted(base["rows"], key=sort_key)}
            if not values_equal(base_unordered, union):
                return {"kind": "tlp", "backends": [b.name], "outcomes": outcomes,
                        "tlp": {"whole": base_unordered, "union": union,
                                "parts": parts}}
    return None


def tlp_valid(case: dict) -> bool:
    """Is the case's TLP predicate well-typed against the pipeline's output?"""
    if case.get("tlp") is None:
        return True
    try:
        schema = validate(case)
        apply_schema({"op": "filter", "pred": case["tlp"]}, schema, None)
        return True
    except InvalidCase:
        return False


def fingerprint(verdict: dict) -> tuple:
    """What must stay the same while shrinking for it to count as 'the same bug'."""
    kind = verdict["kind"]
    if kind == "flaky":
        return (kind, tuple(tuple(g) for g in verdict.get("partition") or [verdict.get("backends", [])]))
    if kind in ("panic", "tlp"):
        return (kind, tuple(verdict["backends"]))
    return (kind, tuple(tuple(g) for g in verdict["partition"]))
