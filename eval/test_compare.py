"""
Tests for the execution-accuracy comparator.

A lenient comparator inflates every score and a strict one deflates every score,
so the comparator needs its own tests before any model number it produces means
anything. Two properties matter:

  Tolerant of cosmetic difference — alias names, column order, join style, and
  the last decimal place of a percentage must all score as correct.

  Strict about semantic difference — a missing volume floor, the wrong status
  string, or the wrong aggregate must score as wrong. These are exactly the
  mistakes a model that has not learned the business rules makes, and a
  comparator that waves them through would report a fine-tune as unnecessary.

Run: python eval/test_compare.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.compare import has_order_by, results_match  # noqa: E402
from warehouse import build_warehouse, execute_query  # noqa: E402

REFERENCE_CCR = (
    "SELECT PROVIDER_ID, COUNT(*) AS total_claims, "
    "ROUND(COUNT(*) FILTER (WHERE ADJUDICATION_STATUS = 'Paid - First Pass') * 100.0 / COUNT(*), 2) "
    "AS clean_claim_rate_pct FROM Fact_Claims_Adjudication GROUP BY PROVIDER_ID "
    "HAVING COUNT(*) >= 20 ORDER BY clean_claim_rate_pct ASC LIMIT 10"
)

# Same answer, different SQL. All of these must score as correct.
EQUIVALENT = {
    "different aliases": (
        "SELECT PROVIDER_ID, COUNT(*) AS n, "
        "ROUND(SUM(CASE WHEN ADJUDICATION_STATUS = 'Paid - First Pass' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS rate "
        "FROM Fact_Claims_Adjudication GROUP BY PROVIDER_ID HAVING COUNT(*) >= 20 ORDER BY rate ASC LIMIT 10"
    ),
    "CTE instead of inline": (
        "WITH base AS (SELECT PROVIDER_ID, COUNT(*) AS total_claims, "
        "ROUND(COUNT(*) FILTER (WHERE ADJUDICATION_STATUS = 'Paid - First Pass') * 100.0 / COUNT(*), 2) AS ccr "
        "FROM Fact_Claims_Adjudication GROUP BY PROVIDER_ID HAVING COUNT(*) >= 20) "
        "SELECT PROVIDER_ID, total_claims, ccr FROM base ORDER BY ccr ASC LIMIT 10"
    ),
}

# Different answer. All of these must score as wrong.
DIVERGENT = {
    "no minimum volume floor": (
        "SELECT PROVIDER_ID, COUNT(*) AS total_claims, "
        "ROUND(COUNT(*) FILTER (WHERE ADJUDICATION_STATUS = 'Paid - First Pass') * 100.0 / COUNT(*), 2) AS ccr "
        "FROM Fact_Claims_Adjudication GROUP BY PROVIDER_ID ORDER BY ccr ASC LIMIT 10"
    ),
    "wrong status string": (
        "SELECT PROVIDER_ID, COUNT(*) AS total_claims, "
        "ROUND(COUNT(*) FILTER (WHERE ADJUDICATION_STATUS = 'Paid') * 100.0 / COUNT(*), 2) AS ccr "
        "FROM Fact_Claims_Adjudication GROUP BY PROVIDER_ID HAVING COUNT(*) >= 20 ORDER BY ccr ASC LIMIT 10"
    ),
    "sorted the wrong way": (
        "SELECT PROVIDER_ID, COUNT(*) AS total_claims, "
        "ROUND(COUNT(*) FILTER (WHERE ADJUDICATION_STATUS = 'Paid - First Pass') * 100.0 / COUNT(*), 2) AS ccr "
        "FROM Fact_Claims_Adjudication GROUP BY PROVIDER_ID HAVING COUNT(*) >= 20 ORDER BY ccr DESC LIMIT 10"
    ),
    "counts denials instead of clean claims": (
        "SELECT PROVIDER_ID, COUNT(*) AS total_claims, "
        "ROUND(COUNT(*) FILTER (WHERE ADJUDICATION_STATUS = 'Denied') * 100.0 / COUNT(*), 2) AS ccr "
        "FROM Fact_Claims_Adjudication GROUP BY PROVIDER_ID HAVING COUNT(*) >= 20 ORDER BY ccr ASC LIMIT 10"
    ),
}


def _run(sql: str) -> list[tuple]:
    return execute_query(sql)[1]


def main() -> int:
    build_warehouse()
    expected = _run(REFERENCE_CCR)
    ordered = has_order_by(REFERENCE_CCR)
    failures = 0

    print("Equivalent queries (must all PASS):")
    for label, sql in EQUIVALENT.items():
        match, reason = results_match(expected, _run(sql), ordered=ordered)
        print(f"  {'PASS' if match else 'FAIL'}  {label}" + ("" if match else f"  <- {reason}"))
        failures += not match

    print("\nDivergent queries (must all be caught):")
    for label, sql in DIVERGENT.items():
        match, _ = results_match(expected, _run(sql), ordered=ordered)
        print(f"  {'FAIL' if match else 'PASS'}  {label}"
              + ("  <- comparator wrongly accepted this" if match else ""))
        failures += match

    print("\nColumn-order insensitivity:")
    swapped = [(c, b, a) for a, b, c in expected]
    match, _ = results_match(expected, swapped, ordered=ordered)
    print(f"  {'PASS' if match else 'FAIL'}  reversed column order still matches")
    failures += not match

    print("\nFloat tolerance:")
    nudged = [(a, b, (c + 1e-9) if isinstance(c, float) else c) for a, b, c in expected]
    match, _ = results_match(expected, nudged, ordered=ordered)
    print(f"  {'PASS' if match else 'FAIL'}  1e-9 difference tolerated")
    failures += not match

    print("\nWindow-function ORDER BY is not result ordering:")
    windowed = "SELECT PROVIDER_ID, ROW_NUMBER() OVER (ORDER BY PROVIDER_ID) AS rn FROM Dim_Provider"
    ok = not has_order_by(windowed)
    print(f"  {'PASS' if ok else 'FAIL'}  OVER (ORDER BY ...) ignored")
    failures += not ok

    print(f"\n{'All comparator tests passed.' if failures == 0 else f'{failures} test(s) failed.'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
