"""
Execution-accuracy evaluation.

Runs each held-out question through one or more backends, executes the
generated SQL against the warehouse, and compares the result set to the
reference query's result set.

Reports three things:

  Execution accuracy   The headline number. Fraction of questions where the
                       generated query ran and returned the reference answer.

  By category          Broken out over paraphrase / composition / novel_shape,
                       because a model can memorise paraphrases of its training
                       set while failing anything requiring composition, and one
                       aggregate number hides that.

  Failure taxonomy     Why the misses missed: no SQL produced, query rejected by
                       the safety layer, SQL error, or ran-but-wrong. The last
                       category is the dangerous one — a query that executes and
                       returns a plausible wrong number is worse than one that
                       crashes, and on a business-metric schema it is the
                       expected failure mode for a model that does not know the
                       definitions.

Usage:
    python eval/run_eval.py --backends gold
    python eval/run_eval.py --backends finetuned base
    python eval/run_eval.py --backends finetuned anthropic:claude-sonnet-4-5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.compare import has_order_by, results_match  # noqa: E402
from models import build_model  # noqa: E402
from warehouse import build_warehouse, execute_query  # noqa: E402

EVAL_PATH = PROJECT_ROOT / "data" / "eval.jsonl"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

FAILURE_KINDS = ("no_sql", "blocked", "sql_error", "wrong_result")


@dataclass
class QuestionResult:
    question: str
    category: str
    generated_sql: str
    correct: bool
    failure: str | None
    detail: str
    latency_ms: float


@dataclass
class BackendResult:
    name: str
    results: list[QuestionResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return sum(r.correct for r in self.results) / len(self.results) if self.results else 0.0

    def accuracy_for(self, category: str) -> tuple[float, int]:
        subset = [r for r in self.results if r.category == category]
        if not subset:
            return 0.0, 0
        return sum(r.correct for r in subset) / len(subset), len(subset)

    def failure_counts(self) -> dict[str, int]:
        counts = dict.fromkeys(FAILURE_KINDS, 0)
        for r in self.results:
            if r.failure:
                counts[r.failure] = counts.get(r.failure, 0) + 1
        return counts

    @property
    def median_latency_ms(self) -> float:
        import statistics

        return statistics.median([r.latency_ms for r in self.results]) if self.results else 0.0


def load_eval_set() -> list[dict]:
    if not EVAL_PATH.exists():
        raise SystemExit(
            f"{EVAL_PATH} not found. Run `python prepare_dataset.py` first."
        )
    with open(EVAL_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(backend_spec: str, questions: list[dict]) -> BackendResult:
    gold_lookup = {q["question"]: q["sql"] for q in questions}
    model = build_model(backend_spec, gold_lookup=gold_lookup)
    out = BackendResult(name=model.name)

    for i, item in enumerate(questions, 1):
        question, reference, category = item["question"], item["sql"], item["category"]

        start = time.perf_counter()
        try:
            generated = model.generate(question)
        except Exception as exc:  # noqa: BLE001
            generated = ""
            print(f"  [{i}/{len(questions)}] generation failed: {exc}")
        latency_ms = (time.perf_counter() - start) * 1000

        result = _score(question, category, generated, reference, latency_ms)
        out.results.append(result)

        mark = "PASS" if result.correct else "FAIL"
        print(f"  [{i}/{len(questions)}] {mark}  {question[:64]}"
              + (f"   ({result.failure})" if result.failure else ""))

    return out


def _score(
    question: str, category: str, generated: str, reference: str, latency_ms: float
) -> QuestionResult:
    def fail(kind: str, detail: str) -> QuestionResult:
        return QuestionResult(question, category, generated, False, kind, detail, latency_ms)

    if not generated.strip():
        return fail("no_sql", "model produced no SQL")

    _, expected_rows = execute_query(reference)

    try:
        _, actual_rows = execute_query(generated)
    except ValueError as exc:
        return fail("blocked", str(exc))
    except Exception as exc:  # noqa: BLE001
        return fail("sql_error", str(exc).splitlines()[0])

    ordered = has_order_by(reference)
    match, reason = results_match(expected_rows, actual_rows, ordered=ordered)
    if not match:
        return fail("wrong_result", reason)

    return QuestionResult(question, category, generated, True, None, "", latency_ms)


def report(backends: list[BackendResult]) -> str:
    lines: list[str] = []
    width = max(len(b.name) for b in backends) + 2

    lines.append("\n" + "=" * (width + 46))
    lines.append("EXECUTION ACCURACY")
    lines.append("=" * (width + 46))
    header = f"{'Backend':<{width}} {'Overall':>9} {'Para':>7} {'Comp':>7} {'Novel':>7} {'p50 ms':>9}"
    lines.append(header)
    lines.append("-" * len(header))
    for b in backends:
        para, _ = b.accuracy_for("paraphrase")
        comp, _ = b.accuracy_for("composition")
        novel, _ = b.accuracy_for("novel_shape")
        lines.append(
            f"{b.name:<{width}} {b.accuracy:>8.1%} {para:>6.0%} {comp:>6.0%} "
            f"{novel:>6.0%} {b.median_latency_ms:>9.0f}"
        )

    lines.append("\nFAILURE TAXONOMY")
    lines.append("-" * len(header))
    lines.append(f"{'Backend':<{width}} " + " ".join(f"{k:>13}" for k in FAILURE_KINDS))
    for b in backends:
        counts = b.failure_counts()
        lines.append(f"{b.name:<{width}} " + " ".join(f"{counts[k]:>13}" for k in FAILURE_KINDS))

    lines.append(
        "\nwrong_result is the category that matters: the query ran and returned a"
        "\nplausible number that was not the right one. On this schema that is what"
        "\nnot knowing the business definitions looks like."
    )

    n = len(backends[0].results) if backends else 0
    lines.append(
        f"\nn = {n} held-out questions. At this size a difference of a few points is"
        "\nnoise; read the category breakdown and the failure mix, not the headline"
        "\nnumber alone."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--backends", nargs="+", default=["gold"],
        help="One or more of: gold, finetuned[:id], base[:id], anthropic[:model]",
    )
    parser.add_argument("--save", action="store_true", help="Write JSON results to eval/results/")
    args = parser.parse_args()

    build_warehouse()
    questions = load_eval_set()
    print(f"Loaded {len(questions)} held-out questions from {EVAL_PATH.name}")

    backends = []
    for spec in args.backends:
        print(f"\nEvaluating: {spec}")
        backends.append(evaluate(spec, questions))

    print(report(backends))

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = RESULTS_DIR / f"eval-{stamp}.json"
        payload = {
            "n_questions": len(questions),
            "backends": [
                {
                    "name": b.name,
                    "accuracy": b.accuracy,
                    "by_category": {
                        c: b.accuracy_for(c)[0]
                        for c in ("paraphrase", "composition", "novel_shape")
                    },
                    "failures": b.failure_counts(),
                    "median_latency_ms": b.median_latency_ms,
                    "questions": [
                        {
                            "question": r.question,
                            "category": r.category,
                            "correct": r.correct,
                            "failure": r.failure,
                            "detail": r.detail,
                            "generated_sql": r.generated_sql,
                        }
                        for r in b.results
                    ],
                }
                for b in backends
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
