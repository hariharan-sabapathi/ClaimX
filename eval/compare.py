"""
Result-set comparison for execution accuracy.

Execution accuracy asks whether the generated query returns the same answer as
the reference query, not whether the SQL text matches. Two queries can differ in
alias names, join order, subquery style, and column order and still be equally
correct — string comparison scores all of that as failure, which makes the
metric useless for judging a model.

The rules below are the ones that matter for this schema:

  Column order       Ignored. `SELECT a, b` and `SELECT b, a` answer the same
                     question. Values are compared as multisets of rows after
                     sorting each row's cells.

  Column names       Ignored. Models pick their own aliases and the reference
                     aliases are arbitrary.

  Row order          Ignored *unless* the reference query has an ORDER BY, in
                     which case order is part of the answer ("worst ten
                     providers" is meaningless unsorted) and is enforced.

  Floats             Compared with a relative tolerance. Percentage KPIs differ
                     in the last decimal depending on where ROUND is applied,
                     and that is not a semantic error.

  Decimals/dates     Normalised to float and ISO string respectively, since
                     DuckDB returns Decimal and date objects whose equality is
                     type-sensitive.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any

REL_TOLERANCE = 1e-4
ABS_TOLERANCE = 1e-6


def _normalise(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        return value
    return str(value)


def _cells_equal(a: Any, b: Any) -> bool:
    a, b = _normalise(a), _normalise(b)
    if a is None or b is None:
        return a is b or a == b
    if isinstance(a, float) and isinstance(b, float):
        if a == b:
            return True
        return abs(a - b) <= max(ABS_TOLERANCE, REL_TOLERANCE * max(abs(a), abs(b)))
    return a == b


def _sort_key(row: tuple) -> tuple:
    """Order-insensitive key that survives mixed types and NULLs."""
    return tuple((v is None, str(type(_normalise(v))), str(_normalise(v))) for v in row)


def _rows_equal(a: tuple, b: tuple) -> bool:
    if len(a) != len(b):
        return False
    return all(_cells_equal(x, y) for x, y in zip(a, b))


def has_order_by(sql: str) -> bool:
    """True when the outermost query orders its results."""
    import re

    # Ignore ORDER BY inside window functions — OVER (ORDER BY ...) does not
    # order the result set.
    stripped = re.sub(r"OVER\s*\([^)]*\)", " ", sql, flags=re.IGNORECASE)
    return bool(re.search(r"\bORDER\s+BY\b", stripped, re.IGNORECASE))


def results_match(
    expected: list[tuple],
    actual: list[tuple],
    *,
    ordered: bool,
) -> tuple[bool, str]:
    """
    Compare two result sets. Returns (match, reason_if_not).
    """
    if len(expected) != len(actual):
        return False, f"row count {len(actual)} != expected {len(expected)}"

    if not expected:
        return True, ""

    if len(expected[0]) != len(actual[0]):
        return False, f"column count {len(actual[0])} != expected {len(expected[0])}"

    # Column order is ignored, so sort each row's cells before comparing.
    exp_rows = [tuple(sorted(r, key=lambda v: (v is None, str(_normalise(v))))) for r in expected]
    act_rows = [tuple(sorted(r, key=lambda v: (v is None, str(_normalise(v))))) for r in actual]

    if not ordered:
        exp_rows = sorted(exp_rows, key=_sort_key)
        act_rows = sorted(act_rows, key=_sort_key)

    for i, (e, a) in enumerate(zip(exp_rows, act_rows)):
        if not _rows_equal(e, a):
            return False, f"row {i} differs: got {a!r}, expected {e!r}"

    return True, ""
