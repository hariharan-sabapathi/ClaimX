"""
The one place the prompt format is defined.

This module exists because of a bug in the inventory version of this project.
There, `prepare_dataset.py` wrote training examples as plain text:

    "Using the inventory database with tables: ... Question: {q}\\n{sql}"

while `inference.py` sent a chat-templated system + user message with a long
instruction block and few-shot examples. The two formats never matched, so at
inference time the model was being asked for something it had never been
trained to produce. Whatever the fine-tune learned was largely bypassed, and
the model was leaning on its base-model ability instead — which is exactly the
thing the project set out to improve on.

So: `build_prompt` is called by prepare_dataset.py, by train.ipynb, and by
inference.py. If the format changes, it changes in one place and everything
that depends on it moves together. The eval harness asserts this.
"""

from __future__ import annotations

from schema import BUSINESS_RULES, SCHEMA_CONTEXT

INSTRUCTION = (
    "You are a DuckDB expert working with a healthcare claims adjudication "
    "warehouse. Write a single SQL SELECT that answers the question. "
    "Output only SQL, no explanation and no markdown."
)


def build_prompt(question: str, include_rules: bool = True) -> str:
    """The exact text the model sees, at training time and at inference time."""
    parts = [INSTRUCTION, "", SCHEMA_CONTEXT]
    if include_rules:
        parts += ["", BUSINESS_RULES.strip()]
    parts += ["", f"Question: {question}", "SQL:"]
    return "\n".join(parts)


def build_training_text(question: str, sql: str, include_rules: bool = True) -> str:
    """Prompt plus target completion, for supervised fine-tuning."""
    return f"{build_prompt(question, include_rules)} {sql.strip()}"


def build_chat_messages(question: str, include_rules: bool = True) -> list[dict]:
    """
    Chat-template form of the same prompt, for base models that expect one.

    The system turn carries the instruction, schema, and rules; the user turn
    carries only the question. Content is identical to build_prompt — just
    partitioned across roles.
    """
    system = [INSTRUCTION, "", SCHEMA_CONTEXT]
    if include_rules:
        system += ["", BUSINESS_RULES.strip()]
    return [
        {"role": "system", "content": "\n".join(system)},
        {"role": "user", "content": f"Question: {question}\nSQL:"},
    ]


def extract_sql(text: str) -> str:
    """
    Pull SQL out of a model completion.

    Handles fenced code blocks, leading prose, and trailing commentary. Returns
    an empty string when nothing resembling a SELECT is present, so callers can
    distinguish "no query" from "bad query" rather than executing prose.
    """
    import re

    text = text.strip()

    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    match = re.search(r"\b(WITH|SELECT)\b", text, re.IGNORECASE)
    if not match:
        return ""

    sql = text[match.start():]

    # Stop at the first statement terminator; models often continue with a
    # second example query or an explanation after the answer.
    if ";" in sql:
        sql = sql.split(";", 1)[0]

    # Drop trailing prose lines that are clearly not SQL.
    lines = []
    for line in sql.splitlines():
        if line.strip().lower().startswith(("question:", "answer:", "explanation:", "note:")):
            break
        lines.append(line)

    return "\n".join(lines).strip()
