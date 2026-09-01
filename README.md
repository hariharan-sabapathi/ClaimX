# ClaimX

**Ask a healthcare claims warehouse questions in plain English. ClaimX uses a domain-specific Text-to-SQL pipeline with QLoRA fine-tuning, read-only SQL execution, and execution-based evaluation.**

![The app answering a question about clean claim rate by provider](assets/demo.png)

## Current status

The end-to-end pipeline is implemented, including the QLoRA fine-tuning workflow, shared inference path, read-only SQL execution, and execution-based evaluation harness.

The repository currently includes the training and evaluation infrastructure. **Model training and full benchmark results have not yet been run or reported.** The results table therefore contains only the `gold` harness check rather than estimated model scores.

ClaimX is designed to test whether a small domain-adapted model can produce **business-correct SQL**, not merely syntactically valid SQL, on a healthcare claims schema.

## Why this project exists

The point of this project is not that a language model can write SQL. Every model can write SQL.

The harder problem is that on a schema like this one, **syntactically valid SQL and correct SQL are different problems**.

Ask a general model:

> "Which providers have the worst clean claim rate?"

It will return something. It may group by provider, count rows, divide, and sort ascending — and still be wrong.

A clean claim in this warehouse specifically means:

`ADJUDICATION_STATUS = 'Paid - First Pass'`

Provider rankings also require a minimum-volume floor:

`HAVING COUNT(*) >= 20`

Without that floor, the top of the league table can fill with providers who submitted three claims and had two denied. That is a 33% clean claim rate, but it is noise rather than a meaningful performance signal.

None of those definitions are obvious from the table structure alone. They live in data dictionaries, dbt models, and the business rules behind the warehouse.

**ClaimX puts those definitions into the Text-to-SQL pipeline.**

## How it works

```text
Plain-English question
        ↓
Schema + healthcare business definitions
        ↓
Qwen2.5-1.5B Text-to-SQL model
        ↓
Generated SQL
        ↓
Read-only execution guard
        ↓
DuckDB claims warehouse
        ↓
Query results
```

The same prompt format is shared between dataset preparation, training, and inference to prevent train/serve skew.

The repository uses **QLoRA supervised fine-tuning (SFT)** to adapt Qwen2.5-1.5B to the claims schema, terminology, relationships, and business definitions.

## Results

| Backend | Overall | Paraphrase | Composition | Novel shape | p50 latency |
|---|---:|---:|---:|---:|---:|
| gold (harness check) | 100.0% | 100% | 100% | 100% | 0 ms |

*Evaluation uses 17 held-out questions across paraphrase, composition, and novel query shapes. Model rows will be populated after an actual evaluation run.*

**Only the `gold` row is currently measured.** The model rows are intentionally absent rather than estimated. There is no honest way to fill them without running the models, and plausible-looking invented numbers would undermine the purpose of the evaluation.

After fine-tuning and configuration, run:

```bash
python eval/run_eval.py --backends gold finetuned base anthropic --save
python eval/update_readme.py
```

The second command rewrites the table from the saved evaluation results so the published numbers remain tied to the measured run.

### What the evaluation is designed to answer

**`gold` must read 100%.** It is an oracle backend that replays the reference SQL. If it scores anything else, the comparator is broken and no model result is trustworthy.

**The baseline is not handicapped.** The base Qwen2.5-1.5B model receives the same schema and the same business-rule definitions as the fine-tuned model. Otherwise, the comparison would simply measure how much information was included in the prompt.

**The larger-model comparison matters.** A reader should reasonably ask why a 1.5B model should run locally when a much larger hosted model is available. Cost, latency, local execution, and keeping a claims schema away from third-party APIs are legitimate reasons. But accuracy should still be measured honestly. If Claude Sonnet wins outright, that belongs in the results.

At **n = 17** held-out questions, small percentage differences should not be overinterpreted. The category breakdown and failure taxonomy provide additional signal.

## Evaluation methodology

ClaimX uses **execution accuracy**, not SQL string similarity.

Generated SQL is executed against the warehouse and compared with the result of the reference query. Two semantically equivalent SQL queries should receive credit even if they differ in aliases, join order, or query structure.

The comparator:

- ignores column order and column names
- ignores row order when ordering is not semantically required
- tolerates floating-point differences in the final decimal
- enforces ordering when the reference query contains `ORDER BY`
- detects known semantic errors

`eval/test_compare.py` tests both directions: cosmetic SQL rewrites must pass, while semantic errors such as dropping the provider volume floor, using the wrong status string, sorting incorrectly, or counting denials instead of clean claims must fail.

### Held-out evaluation set

The 17 evaluation questions are divided into three categories:

| Category | n | What it tests |
|---|---:|---|
| `paraphrase` | 6 | Same question as training, different wording |
| `composition` | 5 | Multiple trained concepts combined |
| `novel_shape` | 6 | Query structures absent from training, including window functions, quantiles, and cohort comparisons |

### Failure taxonomy

| Kind | Meaning |
|---|---|
| `no_sql` | Model produced nothing parseable |
| `blocked` | Query rejected by the read-only guard |
| `sql_error` | Generated SQL failed during execution |
| `wrong_result` | Query executed but returned an incorrect result |

The `wrong_result` category is particularly important.

A query that crashes is annoying. A query that executes successfully and returns a plausible but incorrect number is much more dangerous — especially when the output is being used for business reporting.

## Fine-tuning

ClaimX uses **QLoRA supervised fine-tuning (SFT)** with **Qwen2.5-1.5B**.

The training data contains **158 validated question-SQL pairs** covering:

- the claims schema
- healthcare KPIs
- schema relationships
- domain-specific business definitions

The training workflow is provided in `train.ipynb` and is designed to run on a free T4 GPU.

The goal is not to teach the model SQL syntax from scratch. It is to adapt a small model to the terminology, schema relationships, and business definitions that determine whether a healthcare claims query is actually correct.

## Business definitions

The model needs to understand definitions that are not obvious from the database schema alone:

- A claim is clean when `ADJUDICATION_STATUS = 'Paid - First Pass'`.
- **Denied claims have `BILLED_PAID_AMT = 0` by construction.** Therefore, "how much did denials cost us" cannot be answered by summing that column. It is instead proxied by the average paid amount of first-pass-paid claims of the same claim type.
- Provider league tables require `HAVING COUNT(*) >= 20`.
- Preventable denials are identified with `PREVENTABILITY_BUCKET LIKE 'Preventable%'`.
- Aged A/R means `AR_AGING_BUCKET = '90+'`.

These definitions are part of the domain knowledge ClaimX is designed to encode into the Text-to-SQL system.

CARC codes are real X12 Claim Adjustment Reason Codes. `197` represents missing prior authorization, `18` an exact duplicate, and `29` untimely filing. Their preventability classification is defined by this project's taxonomy.

## Read-only execution

Generated SQL is executed against a DuckDB claims warehouse through a read-only execution layer.

The execution guard prevents generated queries from modifying the warehouse while allowing analytical queries to return results.

The application exposes:

- **Gradio UI** for interactive natural-language querying
- **FastAPI** endpoint for programmatic access

## The schema

ClaimX uses a star schema mirroring the marts in `Medical-Insurance-Claims-And-Denial-Analytics`, with identical table and column names so the Text-to-SQL system can target the same warehouse design.

```text
Dim_Patient(PATIENT_ID, BIRTH_DT, DEATH_DT, SEX, BENE_RACE_CD, SP_STATE_CODE,
            AGE_APPROX, CHRONIC_CONDITION_COUNT, SNAPSHOT_YEAR)

Dim_Provider(PROVIDER_ID, ATTENDING_NPI, CLAIM_VOLUME)

Dim_Diagnosis(DIAGNOSIS_CODE, CODE_SYSTEM, DIAGNOSIS_CATEGORY_APPROX, DESCRIPTION)

Dim_CARC_Denials(CARC_CODE, DESCRIPTION, PREVENTABILITY_BUCKET)

Fact_Claims_Adjudication(CLAIM_ID, PATIENT_ID, PROVIDER_ID, DIAGNOSIS_CODE, CARC_CODE,
                         CLAIM_TYPE, CLM_FROM_DT, CLM_THRU_DT, BILLED_PAID_AMT,
                         PRIMARY_PYR_PD_AMT, ADJUDICATION_STATUS,
                         DAYS_SINCE_SUBMISSION, AR_AGING_BUCKET)
```

## Data: what it is and is not

**Synthetic. No PHI. Denial outcomes are simulated.**

By default, `warehouse.py` generates the warehouse itself:

- 40,000 claims
- 2,666 providers
- 3,333 patients

The distributions are designed to make the business rules meaningful. Provider volume follows a power law with a long tail of low-volume providers, creating the conditions where a minimum-volume threshold matters.

The project can also use CMS DE-SynPUF extracts through `data/raw/`. DE-SynPUF is itself synthetic and does not contain an adjudication field, so denial status remains simulated rather than observed.

**No number produced by this project should be interpreted as describing real payer behaviour.**

The schema, joins, and business definitions are modeled after healthcare claims analytics. The denial outcomes are synthetic.

## Setup

```bash
git clone <this repo> && cd claims-text-to-sql

python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows

pip install -r requirements.txt
cp .env.example .env
```

Build the warehouse and prepare the dataset:

```bash
python warehouse.py
python prepare_dataset.py
```

Validate the evaluation comparator before trusting any evaluation:

```bash
python eval/test_compare.py
```

Run the harness without a model:

```bash
python eval/run_eval.py --backends gold
```

This requires no model and no GPU. It confirms that the evaluation harness is functioning before spending a training run on fine-tuning.

### Fine-tune and evaluate

Fine-tune Qwen2.5-1.5B using `train.ipynb` on a T4 GPU, set `FINETUNED_MODEL` in `.env`, then run:

```bash
python eval/run_eval.py --backends gold finetuned base anthropic --save
python eval/update_readme.py
```

### Launch the application

```bash
python app.py
```

Gradio runs at:

```text
http://127.0.0.1:7860
```

Or launch the API:

```bash
uvicorn api:app --reload
```

FastAPI documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Project structure

```text
schema.py              Star schema DDL, CARC codes, business definitions
warehouse.py           Builds DuckDB; read-only execution guard
prompt.py              Shared prompt format for training and inference
prepare_dataset.py     Generates and validates training/evaluation pairs
train.ipynb             QLoRA supervised fine-tuning workflow
models.py              Model backends behind one interface
inference.py           Text-to-SQL serving path
app.py                 Gradio interface
api.py                 FastAPI read API

data/
  train.jsonl           Training examples
  eval.jsonl            Held-out evaluation questions

eval/
  compare.py            Result-set comparison
  run_eval.py           Evaluation runner and failure taxonomy
  test_compare.py       Comparator tests
  update_readme.py      Updates evaluation results from saved runs
```

## Design decisions

### Shared prompt format

`prompt.py` provides the prompt format used by both training and inference.

This prevents **train/serve skew**, where a model is trained to produce SQL in one format but later asked to generate it using a different prompt structure.

The same prompt function is used across dataset preparation, training, and inference.

### Validated training pairs

`prepare_dataset.py` executes training pairs against the warehouse before they are written to the dataset.

This prevents executable-but-wrong SQL from silently becoming training data.

For example:

```sql
SELECT category, name, MIN(unit_price)
FROM products
GROUP BY category;
```

can execute successfully while returning a `name` from an arbitrary row rather than the product associated with the minimum price.

The dataset preparation step is designed to catch this class of problem before it reaches training.

### Execution-based evaluation

The evaluation system compares query results rather than raw SQL strings.

This allows semantically equivalent SQL to receive credit even when aliases, join order, or query structure differ.

## Why these safeguards matter

Two design decisions were added specifically to avoid mistakes found in the earlier inventory version of this project.

**`prompt.py` prevents train/serve skew.** The earlier version prepared training examples in one format while inference used a different chat-templated structure with few-shot examples. The model was therefore being asked at inference time to produce something different from what it had been trained to produce. A shared prompt function removes that mismatch.

**`prepare_dataset.py` validates every training pair.** The earlier inventory dataset contained executable-but-wrong SQL that could teach the model a semantic error. ClaimX executes each pair against the warehouse before allowing it into the dataset.

The principle is simple:

> **If the evaluation cannot catch the failure, the training pipeline should not silently teach it.**

## Related projects

- `Medical-Insurance-Claims-And-Denial-Analytics` — the underlying healthcare claims warehouse, with PySpark ingestion, dbt marts, and a Power BI denial control tower.
- `clinical-retrieval` — the unstructured counterpart using BM25 retrieval over clinical notes with bootstrap confidence intervals.
- `Finance-Intelligence-Agent` — a contrasting natural-language-to-SQL approach using a hosted model and tool calling rather than fine-tuning.

ClaimX is deliberately different: **the goal is to test whether domain-specific adaptation can make a small local model better at business-correct SQL when the important knowledge lives outside the schema itself.**
