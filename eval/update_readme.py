"""
Fill the README results table from a saved eval run.

Transcribing numbers from terminal output into a README by hand is where
published results quietly drift from measured ones — a digit gets fixed up, a
row from an older run survives a rewrite, and nobody notices because nobody
re-runs it. This reads the JSON that `run_eval.py --save` wrote and rewrites the
table between the two marker comments in README.md.

Usage:
    python eval/run_eval.py --backends gold finetuned base --save
    python eval/update_readme.py                 # uses the most recent run
    python eval/update_readme.py --results eval/results/eval-20260901-200335.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
README = PROJECT_ROOT / "README.md"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"

HEADER = (
    "| Backend | Overall | Paraphrase | Composition | Novel shape | p50 latency |\n"
    "|---|---:|---:|---:|---:|---:|"
)


def latest_results() -> Path:
    runs = sorted(RESULTS_DIR.glob("eval-*.json"))
    if not runs:
        raise SystemExit(
            f"No results in {RESULTS_DIR}. Run `python eval/run_eval.py --backends ... --save` first."
        )
    return runs[-1]


def build_table(payload: dict) -> str:
    rows = [HEADER]
    for b in payload["backends"]:
        cat = b["by_category"]
        rows.append(
            f"| {b['name']} "
            f"| {b['accuracy']:.1%} "
            f"| {cat['paraphrase']:.0%} "
            f"| {cat['composition']:.0%} "
            f"| {cat['novel_shape']:.0%} "
            f"| {b['median_latency_ms']:.0f} ms |"
        )

    measured = ", ".join(b["name"] for b in payload["backends"])
    rows.append(
        f"\n<sub>n = {payload['n_questions']} held-out questions. "
        f"Backends measured in this run: {measured}. "
        f"Regenerate with `python eval/run_eval.py --backends ... --save && "
        f"python eval/update_readme.py`.</sub>"
    )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()

    path = args.results or latest_results()
    payload = json.loads(path.read_text(encoding="utf-8"))

    text = README.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit(f"README.md is missing the {START} / {END} markers.")

    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    README.write_text(
        f"{before}{START}\n\n{build_table(payload)}\n\n{END}{after}", encoding="utf-8"
    )

    print(f"Updated README results table from {path.name}")
    for b in payload["backends"]:
        print(f"  {b['name']:<28} {b['accuracy']:.1%}")


if __name__ == "__main__":
    main()
