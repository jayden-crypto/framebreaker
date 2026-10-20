"""Normalise engine outputs so they can be compared.

Results become (column names, dtype families, rows of canonical values).
Unless the pipeline fixes the order, rows are compared as a multiset.
Ints and floats share one numeric rank so `1` and `1.0` compare equal by value;
dtype differences are tracked separately, since they are a different (and
usually less severe) class of finding.
"""

from __future__ import annotations

import decimal
import math

import pyarrow as pa

REL_TOL = 1e-9
ABS_TOL = 1e-12


def dtype_family(t: pa.DataType) -> str:
    if pa.types.is_integer(t):
        return "int"
    if pa.types.is_floating(t):
        return "float"
    if pa.types.is_decimal(t):
        return "int" if t.scale == 0 else "float"
    if pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_string_view(t):
        return "str"
    if pa.types.is_boolean(t):
        return "bool"
    if pa.types.is_null(t):
        return "null"
    if pa.types.is_dictionary(t):
        return dtype_family(t.value_type)
    return str(t)


def canon(v):
    """Map a Python value to a sortable, comparable tagged tuple."""
    if v is None:
        return (0, 0)
    if isinstance(v, bool):
        return (1, int(v))
    if isinstance(v, int):
        return (2, v)
    if isinstance(v, decimal.Decimal):
        v = int(v) if v == v.to_integral_value() else float(v)
        return (2, v)
    if isinstance(v, float):
        if math.isnan(v):
            return (3, 0)
        return (2, 0.0 if v == 0 else v)  # -0.0 and 0.0 are the same value
    if isinstance(v, str):
        return (4, v)
    return (5, repr(v))


def sort_key(row):
    # Round floats so tiny summation-order differences don't reorder rows.
    return tuple((tag, float(f"{x:.9g}")) if tag == 2 and isinstance(x, float) else (tag, x)
                 for tag, x in row)


def normalize(table: pa.Table, ordered: bool) -> dict:
    cols = [table.column(i).to_pylist() for i in range(table.num_columns)]
    rows = [tuple(canon(v) for v in r) for r in zip(*cols)] if cols else []
    if not ordered:
        rows.sort(key=sort_key)
    return {
        "status": "ok",
        "columns": list(table.column_names),
        "dtypes": [dtype_family(f.type) for f in table.schema],
        "rows": rows,
    }


def _value_eq(a, b) -> bool:
    if a[0] != b[0]:
        return False
    x, y = a[1], b[1]
    if a[0] == 2 and (isinstance(x, float) or isinstance(y, float)):
        if math.isinf(x) or math.isinf(y):
            return x == y
        return math.isclose(x, y, rel_tol=REL_TOL, abs_tol=ABS_TOL)
    return x == y


def values_equal(a: dict, b: dict) -> bool:
    """Compare two ok-outcomes by column names and row values (not dtypes)."""
    if a["columns"] != b["columns"] or len(a["rows"]) != len(b["rows"]):
        return False
    return all(
        len(ra) == len(rb) and all(_value_eq(x, y) for x, y in zip(ra, rb))
        for ra, rb in zip(a["rows"], b["rows"])
    )


def multiset_union(parts: list[dict]) -> dict:
    rows = [r for p in parts for r in p["rows"]]
    rows.sort(key=sort_key)
    return {**parts[0], "rows": rows}


def show_rows(outcome: dict, limit: int = 12) -> str:
    """Human-readable rendering of an outcome for reports."""
    if outcome["status"] != "ok":
        return f"<{outcome['status']}> {outcome.get('error', '')}"
    lines = [" | ".join(f"{c}:{t}" for c, t in zip(outcome["columns"], outcome["dtypes"]))]
    for row in outcome["rows"][:limit]:
        lines.append(" | ".join(_show_value(v) for v in row))
    if len(outcome["rows"]) > limit:
        lines.append(f"... ({len(outcome['rows'])} rows)")
    return "\n".join(lines)


def _show_value(v) -> str:
    tag, x = v
    if tag == 0:
        return "null"
    if tag == 1:
        return "true" if x else "false"
    if tag == 3:
        return "NaN"
    return repr(x)
