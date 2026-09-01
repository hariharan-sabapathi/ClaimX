"""
Inference entry point for the app and the API.

Thin by design: the prompt lives in prompt.py and the backends live in
models.py, so the serving path and the eval path go through identical code.
If they diverged, the number in the README would stop describing the thing
users actually run.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

from models import SQLModel, build_model

load_dotenv()

DEFAULT_BACKEND = os.getenv("BACKEND", "finetuned")


@lru_cache(maxsize=2)
def get_model(backend: str = DEFAULT_BACKEND) -> SQLModel:
    """Load and cache a backend. First call downloads weights and is slow."""
    return build_model(backend)


def generate_sql(question: str, backend: str = DEFAULT_BACKEND) -> str:
    """Natural-language question in, SQL out. Empty string if none produced."""
    if not question or not question.strip():
        return ""
    return get_model(backend).generate(question.strip())


def answer(question: str, backend: str = DEFAULT_BACKEND) -> dict:
    """
    Full round trip: generate, execute, return everything needed to render.

    Always returns the SQL even when execution fails, so the user can see what
    was attempted rather than only that something went wrong.
    """
    from warehouse import execute_query

    result: dict = {"question": question, "sql": "", "columns": [], "rows": [], "error": None}

    try:
        sql = generate_sql(question, backend)
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"Model failed to load or generate: {exc}"
        return result

    result["sql"] = sql
    if not sql:
        result["error"] = "The model did not return SQL. Try rephrasing the question."
        return result

    try:
        columns, rows = execute_query(sql)
        result["columns"], result["rows"] = columns, rows
    except ValueError as exc:
        result["error"] = f"Query blocked: {exc}"
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"SQL error: {str(exc).splitlines()[0]}"

    return result
