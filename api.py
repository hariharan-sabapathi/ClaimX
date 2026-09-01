"""
FastAPI read API.

Follows the same shape as the read API in CMS-Hospital-Performance-Platform:
read-only, typed response models, an explicit health endpoint. Exists so the
fine-tune is consumable by a dashboard or another service rather than only
through the Gradio UI.

Run: uvicorn api:app --reload
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from inference import answer
from warehouse import build_warehouse, execute_query, get_schema_context

load_dotenv()

BACKEND = os.getenv("BACKEND", "finetuned")

app = FastAPI(
    title="Claims Text-to-SQL API",
    description=(
        "Natural-language querying over a CMS-style claims adjudication warehouse. "
        "Read-only. Synthetic data, no PHI."
    ),
    version="1.0.0",
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, examples=["What is our clean claim rate?"])


class QueryResponse(BaseModel):
    question: str
    sql: str
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    error: str | None = None


class SQLRequest(BaseModel):
    sql: str = Field(..., min_length=1, examples=["SELECT COUNT(*) FROM Fact_Claims_Adjudication"])


@app.on_event("startup")
def _startup() -> None:
    build_warehouse()


@app.get("/health")
def health() -> dict:
    columns, rows = execute_query("SELECT COUNT(*) FROM Fact_Claims_Adjudication")
    return {"status": "ok", "claims": rows[0][0], "backend": BACKEND}


@app.get("/schema")
def schema() -> dict:
    return {"schema": get_schema_context()}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Generate SQL from a question, execute it, and return the rows."""
    result = answer(request.question, backend=BACKEND)
    return QueryResponse(
        question=result["question"],
        sql=result["sql"],
        columns=result["columns"],
        rows=[list(r) for r in result["rows"]],
        row_count=len(result["rows"]),
        error=result["error"],
    )


@app.post("/execute", response_model=QueryResponse)
def execute(request: SQLRequest) -> QueryResponse:
    """Execute SQL directly, bypassing the model. Still read-only."""
    try:
        columns, rows = execute_query(request.sql)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc).splitlines()[0]) from exc

    return QueryResponse(
        question="",
        sql=request.sql,
        columns=columns,
        rows=[list(r) for r in rows],
        row_count=len(rows),
    )
