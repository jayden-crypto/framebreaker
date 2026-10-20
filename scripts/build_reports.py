"""Build ready-to-file bug report drafts from reports/repros/*.py.

Each report embeds its reproducer *and the output of actually running it* on the
installed versions, so the drafts can never drift from reality. Re-run after a
library upgrade to see which bugs are fixed.
"""

from __future__ import annotations

import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPROS = ROOT / "reports" / "repros"

REPORTS = [
    dict(
        file="polars-scalar-column-literal-cmp", repro="polars_scalar_column_literal_cmp.py",
        repo="pola-rs/polars",
        title="`lit(x) <cmp> col` returns a length-1 result when the column was created from a literal (regression in 1.14.0)",
        summary=(
            "If a DataFrame column was created from a literal (e.g. `with_columns(s=pl.lit(1))`), "
            "comparing it with a literal **on the left-hand side** (`pl.lit(0) == pl.col('s')`) returns a "
            "single row instead of one row per row of the frame. In `select` the result is silently "
            "truncated; next to other columns it raises `InvalidOperationError`. Swapping the operands, "
            "arithmetic (`pl.lit(0) + pl.col('s')`) and the streaming engine are all correct."),
        expected="`pl.lit(0) == pl.col('s')` has the frame's height (3), like `pl.col('s') == pl.lit(0)`.",
        notes=(
            "- Affects `==`, `!=`, `<`, `<=`, `>`, `>=` with any literal dtype (int, bool, str), eager and lazy, "
            "with or without optimizations; the streaming engine is correct.\n"
            "- Bisected: correct in 1.13.1, broken from 1.14.0 through 1.44.2.\n"
            "- Possibly related to #29644 (`clip` with a length-1 input), but that issue states other "
            "scalar-receiver ops broadcast correctly; comparisons against a scalar-backed column don't."),
    ),
    dict(
        file="polars-median-neg-inf", repro="polars_median_neg_inf.py",
        repo="pola-rs/polars",
        title="`median` / `quantile(interpolation='linear')` returns NaN when the lower middle value is -inf",
        summary=(
            "With an even number of values whose lower middle value is `-inf`, `median()` (and "
            "`quantile(0.5)` with linear interpolation) returns `NaN`. The symmetric case with `+inf` as the "
            "upper middle value correctly returns `inf`. NumPy, DuckDB and pandas return `-inf`. "
            "Same result in `select`, `group_by().agg()` and the streaming engine; "
            "`quantile(0.5, interpolation='midpoint')` correctly returns `-inf`."),
        expected="`-inf` (the mean of the two middle values, `(-inf + 0.0) / 2`).",
        notes=(
            "- Likely cause: linear interpolation computed as `lo + (hi - lo) * t`. With `lo = -inf` and finite "
            "`hi`, `hi - lo = inf` and `-inf + inf = NaN`. Computing `lo * (1 - t) + hi * t`, or returning `lo` "
            "when `lo == hi` or `lo` is infinite, avoids it.\n"
            "- Found because the differential fuzzer saw Polars alone disagree with DuckDB, PyArrow and pandas "
            "on a grouped median."),
    ),
    dict(
        file="duckdb-arrow-pushdown-float-semantics", repro="duckdb_arrow_pushdown_float_semantics.py",
        repo="duckdb/duckdb-python",
        title="Wrong results: filters pushed into Arrow scans use Arrow float semantics (NaN, -0.0) instead of DuckDB's",
        summary=(
            "When a filter is pushed down into an Arrow scan (a registered pyarrow Table, or a Polars "
            "DataFrame via replacement scan), it is evaluated with Arrow/IEEE semantics instead of DuckDB's. "
            "Rows then silently disappear:\n\n"
            "- `WHERE x > 0` drops NaN rows, although in DuckDB NaN is greater than every number, and "
            "`SELECT x > 0` on the same table returns `true` for them.\n"
            "- `WHERE x IN (0.0, 7.0)` drops `-0.0` rows, although `-0.0 = 0.0` in DuckDB.\n"
            "- The dynamic filters created by join filter pushdown have the same problem, so plain equi-joins "
            "on float keys lose matches between `-0.0` and `0.0`.\n\n"
            "The same queries on native DuckDB tables (and on pandas DataFrames) are correct. A query's "
            "result therefore depends on whether the optimizer decides to push a filter down."),
        expected="The same rows as on a native table: `[(1.0,), (nan,)]` and the `-0.0` row respectively.",
        notes=(
            "- Disabling `filter_pushdown` / `join_filter_pushdown` restores correct results.\n"
            "- Likely cause: pushed-down filters are translated to `pyarrow.compute` expressions. Arrow "
            "comparisons follow IEEE (NaN compares false) and `is_in` hashes -0.0 and 0.0 differently "
            "(apache/arrow#26214): `pc.is_in(pa.array([-0.0]), value_set=pa.array([0.0]))` is `[False]`.\n"
            "- Possible fixes: don't push float comparison/IN filters into Arrow scans, or keep them as "
            "a residual DuckDB filter (push down a superset), or normalise NaN/-0.0 when translating.\n"
            "- Related but different symptoms in the same code path: #615 (NULL rows pass a pushed-down "
            "`<= 'nan'` filter) and duckdb/duckdb#16942 (filtering *for* NaN in a Polars scan). This report covers "
            "NaN *column values* against ordinary constants, and -0.0 in IN-lists and join filters."),
    ),
    dict(
        file="duckdb-python-nan-constant", repro="duckdb_python_nan_constant.py",
        repo="duckdb/duckdb-python",
        title="`ConstantExpression(float('nan'))` still produces NULL instead of NaN (bind parameters were fixed in duckdb#4623)",
        summary=(
            "A NaN passed through the relational API becomes SQL NULL, while `'nan'::DOUBLE` in SQL and "
            "`float('inf')` via `ConstantExpression` keep their values. Libraries that build queries through "
            "the expression API (e.g. Narwhals' `nw.lit(float('nan'))` on DuckDB, which renders as "
            "`CAST(NULL AS DOUBLE)`) therefore turn NaN into NULL, which then behaves differently under "
            "`IS NULL`, `count`, and comparisons."),
        expected="`ConstantExpression(float('nan'))` evaluates to NaN (a DOUBLE), like `'nan'::DOUBLE`.",
        notes=(
            "- duckdb/duckdb#4623 reported the same NaN-to-NULL conversion for bind parameters; that path now "
            "preserves NaN (`con.execute('select ?::DOUBLE', [float('nan')])` returns `nan`), but the relational "
            "API's `ConstantExpression` still converts it.\n"
            "- If the conversion is intentional here (pandas-style missing values), documenting it would help."),
    ),
    dict(
        file="pandas-merge-bool-mixed-backend", repro="pandas_merge_bool_mixed_backend.py",
        repo="pandas-dev/pandas",
        title="BUG: merge on numpy bool vs bool[pyarrow] key containing NA raises ArrowInvalid",
        summary=(
            "Merging a numpy `bool` key with a `bool[pyarrow]` key that contains a missing value raises "
            "`ArrowInvalid: Could not convert 1 with type int: tried to convert to boolean`. Without the NA, "
            "or with `bool[pyarrow]` on both sides, the merge works."),
        expected="The merge succeeds (the NA key simply doesn't match), returning an empty frame here.",
        notes=(
            "- Traceback goes through `merge._factorize_keys`, which for `ArrowExtensionArray` calls "
            "`lk.to_numpy(na_value=1, dtype=lk.dtype.numpy_dtype)` under the comment \"we can only get here with "
            "numeric dtypes\". Boolean Arrow arrays do reach it, and `1` can't be boxed as an Arrow boolean. "
            "Using a dtype-appropriate placeholder (e.g. `True` for bool) would fix it."),
    ),
    dict(
        file="narwhals-pandas-to-arrow-index", repro="narwhals_pandas_to_arrow_index.py",
        repo="narwhals-dev/narwhals",
        title="[Bug]: `DataFrame.to_arrow()` on pandas leaks the index as `__index_level_0__` after a filter",
        summary=(
            "For pandas-backed frames, `to_arrow()` calls `pa.Table.from_pandas(self.native)` without "
            "`preserve_index=False`. Whether an extra `__index_level_0__` column appears then depends on "
            "*which rows* a previous filter kept: a contiguous index is stored as metadata, a non-contiguous "
            "one becomes a column. The output schema depends on the data, and doesn't match other backends."),
        expected="`to_arrow()` returns only the frame's columns, as for every other backend.",
        notes="- One-line fix: `pa.Table.from_pandas(self.native, preserve_index=False)` (the cuDF branch already does this). Still present on `main`.",
    ),
    dict(
        file="narwhals-pandas-literal-dtypes", repro="narwhals_pandas_literal_dtypes.py",
        repo="narwhals-dev/narwhals",
        title="[Bug]: pandas-like `lit()` mishandles nulls and dtypes (`lit(None, Boolean)` becomes False)",
        summary=(
            "On the pandas backend, literals are built with `from_iterable([value])` and then cast, which "
            "goes through numpy dtypes regardless of the frame's dtype backend:\n\n"
            "1. `nw.lit(None, dtype=nw.Boolean)` silently becomes `False` (wrong data).\n"
            "2. `nw.lit(None, dtype=nw.Int64)` raises `TypeError`.\n"
            "3. On pyarrow-backed frames, a string literal in `when/then/otherwise` produces pandas 3's "
            "NaN-semantics `str` dtype, so null propagation is lost (`str.starts_with` on a null returns "
            "`False` instead of null)."),
        expected="Polars semantics: typed null literals stay null, and literal dtypes follow the frame's backend.",
        notes="- 1 and 2 also happen on numpy-backed frames. 3 also makes joins against `bool[pyarrow]` keys fail (pandas then hits a separate merge bug).",
    ),
    dict(
        file="narwhals-duckdb-n-unique-empty", repro="narwhals_duckdb_n_unique_empty.py",
        repo="narwhals-dev/narwhals",
        title="[Bug]: `n_unique()` returns NULL instead of 0 on an empty DuckDB relation",
        summary=(
            "The DuckDB backend renders `n_unique` as `count(DISTINCT x) + max(CASE WHEN x IS NULL THEN 1 "
            "ELSE 0 END)`. Over zero rows `max()` is NULL, so the whole expression is NULL instead of 0."),
        expected="`0`, as in Polars, pandas and PyArrow.",
        notes="- Wrapping the second term in `coalesce(..., 0)` (or using `count(*) > count(x)`) fixes it. The same expression is used for `n_unique().over(...)`.",
    ),
    dict(
        file="narwhals-str-slice-negative-offset", repo="narwhals-dev/narwhals",
        repro="narwhals_str_slice_negative_offset.py",
        title="[Bug]: `str.slice` with a negative offset gives wrong results on DuckDB, PyArrow and pandas",
        summary=(
            "Compared with Polars (the reference), `str.slice(offset, length)` with a negative offset is "
            "wrong in two different ways:\n\n"
            "- **PyArrow / pandas**: when `offset + length >= 0` the stop index is computed as `offset + "
            "length` (e.g. 0), giving an empty string instead of slicing to the end.\n"
            "- **DuckDB**: when `|offset| > len(s)` the generated `substr(c, length(c) + offset, ...)` "
            "starts at a non-positive position, giving the wrong characters (`'abc'.slice(-5, 3)` returns `'c'`)."),
        expected="Polars: the start clamps to 0 and the slice covers `[len+offset, len+offset+length)` intersected with the string.",
        notes="",
    ),
    dict(
        file="narwhals-pyarrow-ends-with-empty", repro="narwhals_pyarrow_ends_with_empty.py",
        repo="narwhals-dev/narwhals",
        title="[Bug]: `str.ends_with('')` returns False for non-empty strings on PyArrow",
        summary=(
            "Every string ends with the empty string, and PyArrow's own `pc.ends_with(..., pattern='')` "
            "returns True. Narwhals' PyArrow backend compares `utf8_slice_codeunits(s, start=-len(suffix))` "
            "with the suffix; for an empty suffix `-0 == 0` slices the *whole* string, which only equals `''` "
            "for empty strings."),
        expected="`[True, True, True]`, as for `starts_with('')` and the other backends.",
        notes="- Using `pc.ends_with(native, pattern=suffix)` directly would fix it.",
    ),
    dict(
        file="narwhals-str-method-literal-receiver", repro="narwhals_str_method_literal_receiver.py",
        repo="narwhals-dev/narwhals",
        title="[Bug]: string methods on a literal receiver fail on PyArrow and pandas",
        summary=(
            "`nw.lit('x').str.contains(...)`, `.starts_with(...)` and `.ends_with(...)` raise `TypeError: "
            "... only supports str pattern values` on PyArrow and pandas, although the pattern *is* a str: "
            "when the receiver is itself a scalar, the extracted pattern is an array rather than a scalar. On "
            "PyArrow, `nw.lit('x').str.len_chars().fill_null(0)` raises `AttributeError: 'ChunkedArray' object "
            "has no attribute 'as_py'`."),
        expected="Polars: the expression broadcasts like any other scalar expression.",
        notes="- Same family as #3964 (scalar-like receiver for `clip`), but a different code path.",
    ),
    dict(
        file="narwhals-agg-null-dtype", repro="narwhals_agg_null_dtype.py",
        repo="narwhals-dev/narwhals",
        title="[Bug]: aggregations whose result is null come back null-typed on PyArrow",
        summary=(
            "On PyArrow, a scalar aggregation that evaluates to null (e.g. `std` of one value, or `min` of an "
            "empty column) produces a column of Arrow `null` type (`Unknown` in Narwhals) instead of keeping "
            "the aggregation's dtype. Downstream operations then fail: `fill_null(0.0)` raises `ArrowInvalid: "
            "Invalid null value`, and a later `group_by(...).agg(std)` raises `ArrowNotImplementedError` for "
            "input type null. `pc.min` itself returns a typed `DoubleScalar`, so the type is lost inside Narwhals."),
        expected="`Float64`, as with Polars (and as with pandas for this case).",
        notes="",
    ),
]


def run(repro: Path) -> str:
    proc = subprocess.run([sys.executable, str(repro)], capture_output=True, text=True, cwd=ROOT)
    out = proc.stdout
    if proc.returncode:
        # keep the exception line (and any hint after it), not the whole traceback
        lines = proc.stderr.strip().splitlines()
        starts = [i for i, line in enumerate(lines)
                  if re.match(r"^[\w.]+(Error|Exception|Invalid)\b.*:", line)]
        out += "\n".join(lines[starts[-1]:] if starts else lines[-1:]) + "\n"
    return out.strip()


def versions() -> str:
    libs = ("polars", "pandas", "pyarrow", "duckdb", "narwhals")
    return ", ".join(f"{lib} {metadata.version(lib)}" for lib in libs) + f", Python {sys.version.split()[0]}"


def main() -> None:
    index = ["# Bug reports", "",
             "Drafts generated by `scripts/build_reports.py`: each embeds its reproducer and the output of",
             "running it on the versions below. **Status: not yet filed.** Search the tracker once more,",
             "then file from your own GitHub account.", "",
             f"Versions: {versions()}", "",
             "| # | Project | Issue | Draft |", "|---|---|---|---|"]
    for i, r in enumerate(REPORTS, 1):
        repro = REPROS / r["repro"]
        code = repro.read_text().strip()
        output = run(repro)
        body = [
            f"<!-- file at https://github.com/{r['repo']}/issues/new -->",
            f"# {r['title']}", "",
            "## Description", "", r["summary"], "",
            "## Reproducible example", "", "```python", code, "```", "",
            "## Output", "", "```", output, "```", "",
            "## Expected behavior", "", r["expected"], "",
        ]
        if r["notes"]:
            body += ["## Notes", "", r["notes"], ""]
        body += ["## Versions", "", versions(), "",
                 "---", "Found with differential fuzzing (Polars/pandas/DuckDB/PyArrow via Narwhals) and "
                 "minimised automatically.", ""]
        (ROOT / "reports" / f"{r['file']}.md").write_text("\n".join(body))
        index.append(f"| {i} | `{r['repo']}` | {r['title']} | [{r['file']}.md]({r['file']}.md) |")
        print(f"wrote reports/{r['file']}.md")
    (ROOT / "reports" / "README.md").write_text("\n".join(index) + "\n")


if __name__ == "__main__":
    main()
