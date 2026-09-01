"""Gradio UI for the claims text-to-SQL model."""

from __future__ import annotations

import os

import gradio as gr
import pandas as pd
from dotenv import load_dotenv

from inference import answer
from warehouse import build_warehouse, summarise

load_dotenv()

BACKEND = os.getenv("BACKEND", "finetuned")

EXAMPLE_QUESTIONS = [
    "What is our clean claim rate?",
    "Which providers have the worst clean claim rate?",
    "What are the top denial reasons by estimated financial loss?",
    "How much revenue are we losing to preventable denials?",
    "How much is sitting in aged A/R?",
    "Show the preventable versus non-preventable denial mix.",
    "Compare inpatient and outpatient performance.",
    "For each aging bucket, what is the single most common denial reason?",
]


def process(question: str) -> tuple[str, pd.DataFrame | None, str]:
    if not question or not question.strip():
        return "", None, "Enter a question."

    result = answer(question, backend=BACKEND)
    sql = result["sql"]

    if result["error"]:
        return sql, None, result["error"]

    df = pd.DataFrame(result["rows"], columns=result["columns"]) if result["columns"] else pd.DataFrame()
    note = f"{len(df)} row(s)."
    if len(df) == 500:
        note += " Output capped at 500 rows."
    return sql, df, note


def create_app() -> gr.Blocks:
    with gr.Blocks(title="Claims Text-to-SQL") as demo:
        gr.Markdown(
            "# Claims Text-to-SQL\n"
            "Ask questions about a healthcare claims adjudication warehouse in plain English. "
            "A fine-tuned model writes DuckDB SQL against the CMS-style star schema and the "
            "query runs read-only against the warehouse.\n\n"
            "**Synthetic data, no PHI.** Denial outcomes are simulated — see the README for "
            "what that means for any number you see here."
        )

        question_input = gr.Textbox(
            label="Question",
            placeholder="e.g. Which providers have the worst clean claim rate?",
            lines=2,
        )

        with gr.Row():
            submit_btn = gr.Button("Run query", variant="primary")
            clear_btn = gr.Button("Clear")

        gr.Examples(examples=[[q] for q in EXAMPLE_QUESTIONS], inputs=question_input)

        sql_output = gr.Code(label="Generated SQL", language="sql", lines=6)
        status_output = gr.Textbox(label="Status", interactive=False)
        results_output = gr.Dataframe(label="Results", wrap=True)

        with gr.Accordion("Warehouse", open=False):
            gr.Markdown(f"```\n{summarise()}\n```")

        for trigger in (submit_btn.click, question_input.submit):
            trigger(
                fn=process,
                inputs=question_input,
                outputs=[sql_output, results_output, status_output],
            )

        clear_btn.click(
            fn=lambda: ("", "", None, ""),
            outputs=[question_input, sql_output, results_output, status_output],
        )

    return demo


if __name__ == "__main__":
    build_warehouse()
    create_app().launch()
