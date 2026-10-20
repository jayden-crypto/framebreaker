"""Test-case reduction.

Greedy delta debugging over the IR: repeatedly try smaller or simpler variants
(drop an op, drop rows, drop a column, replace a sub-expression with a child or
a literal, replace a value with a boring one) and keep any variant that still
fails *in the same way*. A 3,000-row, 5-op failure typically ends up as one or
two rows and one op, which is what a maintainer wants in a bug report.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Iterator

from .ir import BOOL, EXPR_KINDS, FLOAT, INT, STR, InvalidCase, validate

SIMPLE_VALUES = {INT: 0, FLOAT: 0.0, STR: "", BOOL: False}
SIMPLE_LITERALS = [{"op": "lit", "value": v, "dtype": t} for t, v in SIMPLE_VALUES.items()] + [
    {"op": "lit", "value": None, "dtype": t} for t in SIMPLE_VALUES
]
CHILD_KEYS = ("l", "r", "x", "cond", "then", "otherwise")


def _get(node: Any, path: tuple) -> Any:
    for p in path:
        node = node[p]
    return node


def _replaced(case: dict, path: tuple, value: Any) -> dict:
    new = copy.deepcopy(case)
    parent = _get(new, path[:-1])
    parent[path[-1]] = value
    return new


def _expr_paths(node: Any, path: tuple = ()) -> Iterator[tuple]:
    if isinstance(node, dict):
        if node.get("op") in EXPR_KINDS:
            yield path
        for k, v in node.items():
            yield from _expr_paths(v, path + (k,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _expr_paths(v, path + (i,))


def _without(seq: list, start: int, stop: int) -> list:
    return seq[:start] + seq[stop:]


def candidates(case: dict) -> Iterator[dict]:
    pipe = case["pipeline"]

    # 1. drop whole operations
    for i in range(len(pipe)):
        if len(pipe) > 1:
            yield {**case, "pipeline": _without(pipe, i, i + 1)}

    # 2. drop the TLP predicate (only survives if the bug isn't TLP-specific)
    if case.get("tlp") is not None:
        yield {k: v for k, v in case.items() if k != "tlp"}

    # 3. drop the right-hand table if no join uses it
    if len(case["tables"]) > 1 and not any(op["op"] == "join" for op in pipe):
        yield {**case, "tables": case["tables"][:1]}

    # 4. drop rows, big chunks first (ddmin)
    for ti, t in enumerate(case["tables"]):
        rows = t["rows"]
        chunk = len(rows) // 2 or 1
        while rows and chunk >= 1:
            for start in range(0, len(rows), chunk):
                tables = list(case["tables"])
                tables[ti] = {**t, "rows": _without(rows, start, start + chunk)}
                yield {**case, "tables": tables}
            chunk //= 2

    # 5. drop columns
    for ti, t in enumerate(case["tables"]):
        names = list(t["columns"])
        if len(names) > 1:
            for ci, name in enumerate(names):
                tables = list(case["tables"])
                tables[ti] = {
                    "columns": {c: d for c, d in t["columns"].items() if c != name},
                    "rows": [r[:ci] + r[ci + 1:] for r in t["rows"]],
                }
                yield {**case, "tables": tables}

    # 6. simplify operation arguments
    for i, op in enumerate(pipe):
        for key in ("aggs", "keys", "cols", "on"):
            items = op.get(key) or []
            if len(items) > 1:
                for j in range(len(items)):
                    new_op = {**op, key: _without(items, j, j + 1)}
                    yield {**case, "pipeline": pipe[:i] + [new_op] + pipe[i + 1:]}
        if op["op"] == "sort" and len(op["by"]) > 1:
            for j in range(len(op["by"])):
                new_op = {**op, "by": _without(op["by"], j, j + 1),
                          "descending": _without(op["descending"], j, j + 1)}
                yield {**case, "pipeline": pipe[:i] + [new_op] + pipe[i + 1:]}
        if op["op"] == "drop_nulls" and op["subset"]:
            yield {**case, "pipeline": pipe[:i] + [{**op, "subset": None}] + pipe[i + 1:]}

    # 7. simplify expressions: replace with a child, then with a literal
    for path in _expr_paths(case):
        e = _get(case, path)
        for k in CHILD_KEYS:
            if k in e:
                yield _replaced(case, path, copy.deepcopy(e[k]))
        if e["op"] not in ("lit", "col"):
            for lit in SIMPLE_LITERALS:
                yield _replaced(case, path, dict(lit))
        if e["op"] == "lit" and e["value"] is not None and e["value"] != SIMPLE_VALUES[e["dtype"]]:
            yield _replaced(case, path, {**e, "value": SIMPLE_VALUES[e["dtype"]]})

    # 8. replace data values with boring ones
    for ti, t in enumerate(case["tables"]):
        dtypes = list(t["columns"].values())
        for ri, row in enumerate(t["rows"]):
            for ci, v in enumerate(row):
                simple = SIMPLE_VALUES[dtypes[ci]]
                if v is not None and (v != simple or str(v) != str(simple)):
                    yield _replaced(case, ("tables", ti, "rows", ri, ci), simple)


def shrink(case: dict, still_fails: Callable[[dict], bool], is_valid: Callable[[dict], bool],
           budget: int = 3000) -> tuple[dict, int]:
    """Return (smallest failing case found, number of oracle evaluations)."""
    current = case
    evals = 0
    progress = True
    while progress and evals < budget:
        progress = False
        for cand in candidates(current):
            try:
                validate(cand)
            except InvalidCase:
                continue
            if not is_valid(cand):
                continue
            evals += 1
            if still_fails(cand):
                current = cand
                progress = True
                break
            if evals >= budget:
                break
    return current, evals
