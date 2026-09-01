"""
Builds the training and evaluation sets.

Design notes:

1. Every pair is validated by executing its SQL against the warehouse before it
   is written out. A training set with queries that do not run teaches the model
   to produce queries that do not run. The inventory version of this project
   skipped this step and shipped a `MIN(unit_price)` GROUP BY that returns a
   name from an arbitrary row.

2. The hand-written pairs come first and carry the weight. They encode the KPI
   definitions from the source warehouse — clean claim rate, preventability
   mix, A/R aging, the minimum-volume floor on provider rankings, the
   average-paid proxy for denied dollars. These are the queries a general model
   gets semantically wrong even when handed the schema, and they are the
   argument for fine-tuning at all.

3. Templated pairs fill out coverage of the value space (CARC codes, claim
   types, aging buckets, states) but are deliberately the minority. A dataset
   that is mostly template is a dataset that teaches one query shape.

4. The eval split is held out by *question*, not by row, and includes
   paraphrases of training questions plus questions whose SQL shape never
   appears in training. Reporting accuracy on questions the model was trained
   on measures memorisation.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from prompt import build_training_text
from schema import AR_AGING_BUCKETS, CARC_CODES, CLAIM_TYPES, PREVENTABILITY_BUCKETS
from warehouse import build_warehouse, execute_query

DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_PATH = DATA_DIR / "train.jsonl"
EVAL_PATH = DATA_DIR / "eval.jsonl"

# ---------------------------------------------------------------------------
# Tier 1 — business definitions. The core of the dataset.
# ---------------------------------------------------------------------------
BUSINESS_PAIRS: list[tuple[str, str]] = [
    (
        "What is our clean claim rate?",
        "SELECT ROUND(COUNT(*) FILTER (WHERE ADJUDICATION_STATUS = 'Paid - First Pass') * 100.0 / COUNT(*), 2) AS clean_claim_rate_pct FROM Fact_Claims_Adjudication",
    ),
    (
        "What is the denial rate overall?",
        "SELECT ROUND(COUNT(*) FILTER (WHERE ADJUDICATION_STATUS = 'Denied') * 100.0 / COUNT(*), 2) AS denial_rate_pct FROM Fact_Claims_Adjudication",
    ),
    (
        "Show clean claim rate by provider.",
        "SELECT PROVIDER_ID, COUNT(*) AS total_claims, ROUND(COUNT(*) FILTER (WHERE ADJUDICATION_STATUS = 'Paid - First Pass') * 100.0 / COUNT(*), 2) AS clean_claim_rate_pct FROM Fact_Claims_Adjudication GROUP BY PROVIDER_ID HAVING COUNT(*) >= 20 ORDER BY clean_claim_rate_pct ASC",
    ),
    (
        "Which providers have the worst clean claim rate?",
        "SELECT PROVIDER_ID, COUNT(*) AS total_claims, ROUND(COUNT(*) FILTER (WHERE ADJUDICATION_STATUS = 'Paid - First Pass') * 100.0 / COUNT(*), 2) AS clean_claim_rate_pct FROM Fact_Claims_Adjudication GROUP BY PROVIDER_ID HAVING COUNT(*) >= 20 ORDER BY clean_claim_rate_pct ASC LIMIT 10",
    ),
    (
        "Which providers have the best clean claim rate?",
        "SELECT PROVIDER_ID, COUNT(*) AS total_claims, ROUND(COUNT(*) FILTER (WHERE ADJUDICATION_STATUS = 'Paid - First Pass') * 100.0 / COUNT(*), 2) AS clean_claim_rate_pct FROM Fact_Claims_Adjudication GROUP BY PROVIDER_ID HAVING COUNT(*) >= 20 ORDER BY clean_claim_rate_pct DESC LIMIT 10",
    ),
    (
        "What are the top denial reasons by volume?",
        "SELECT f.CARC_CODE, c.DESCRIPTION, COUNT(*) AS denial_count FROM Fact_Claims_Adjudication f JOIN Dim_CARC_Denials c ON f.CARC_CODE = c.CARC_CODE WHERE f.ADJUDICATION_STATUS = 'Denied' GROUP BY f.CARC_CODE, c.DESCRIPTION ORDER BY denial_count DESC LIMIT 10",
    ),
    (
        "What are the top denial reasons by estimated financial loss?",
        "SELECT f.CARC_CODE, c.DESCRIPTION, COUNT(*) AS denial_count, ROUND(COUNT(*) * (SELECT AVG(BILLED_PAID_AMT) FROM Fact_Claims_Adjudication WHERE ADJUDICATION_STATUS = 'Paid - First Pass'), 2) AS estimated_financial_loss FROM Fact_Claims_Adjudication f JOIN Dim_CARC_Denials c ON f.CARC_CODE = c.CARC_CODE WHERE f.ADJUDICATION_STATUS = 'Denied' GROUP BY f.CARC_CODE, c.DESCRIPTION ORDER BY estimated_financial_loss DESC",
    ),
    (
        "How much revenue are we losing to preventable denials?",
        "SELECT ROUND(COUNT(*) * (SELECT AVG(BILLED_PAID_AMT) FROM Fact_Claims_Adjudication WHERE ADJUDICATION_STATUS = 'Paid - First Pass'), 2) AS estimated_financial_loss FROM Fact_Claims_Adjudication f JOIN Dim_CARC_Denials c ON f.CARC_CODE = c.CARC_CODE WHERE f.ADJUDICATION_STATUS = 'Denied' AND c.PREVENTABILITY_BUCKET LIKE 'Preventable%'",
    ),
    (
        "Show the preventable versus non-preventable denial mix.",
        "SELECT c.PREVENTABILITY_BUCKET, COUNT(*) AS denial_count FROM Fact_Claims_Adjudication f JOIN Dim_CARC_Denials c ON f.CARC_CODE = c.CARC_CODE WHERE f.ADJUDICATION_STATUS = 'Denied' GROUP BY c.PREVENTABILITY_BUCKET ORDER BY denial_count DESC",
    ),
    (
        "What proportion of denials are preventable?",
        "SELECT ROUND(COUNT(*) FILTER (WHERE c.PREVENTABILITY_BUCKET LIKE 'Preventable%') * 100.0 / COUNT(*), 2) AS preventable_pct FROM Fact_Claims_Adjudication f JOIN Dim_CARC_Denials c ON f.CARC_CODE = c.CARC_CODE WHERE f.ADJUDICATION_STATUS = 'Denied'",
    ),
    (
        "Show the A/R aging matrix by provider.",
        "WITH avg_paid AS (SELECT CLAIM_TYPE, AVG(BILLED_PAID_AMT) AS avg_paid_amt FROM Fact_Claims_Adjudication WHERE ADJUDICATION_STATUS = 'Paid - First Pass' GROUP BY CLAIM_TYPE) SELECT f.PROVIDER_ID, f.AR_AGING_BUCKET, COUNT(*) AS claim_count, ROUND(SUM(a.avg_paid_amt), 2) AS at_risk_amt_proxy FROM Fact_Claims_Adjudication f JOIN avg_paid a ON f.CLAIM_TYPE = a.CLAIM_TYPE WHERE f.ADJUDICATION_STATUS = 'Denied' GROUP BY f.PROVIDER_ID, f.AR_AGING_BUCKET ORDER BY at_risk_amt_proxy DESC",
    ),
    (
        "How much is sitting in aged A/R?",
        "WITH avg_paid AS (SELECT CLAIM_TYPE, AVG(BILLED_PAID_AMT) AS avg_paid_amt FROM Fact_Claims_Adjudication WHERE ADJUDICATION_STATUS = 'Paid - First Pass' GROUP BY CLAIM_TYPE) SELECT ROUND(SUM(a.avg_paid_amt), 2) AS at_risk_amt_proxy, COUNT(*) AS claim_count FROM Fact_Claims_Adjudication f JOIN avg_paid a ON f.CLAIM_TYPE = a.CLAIM_TYPE WHERE f.ADJUDICATION_STATUS = 'Denied' AND f.AR_AGING_BUCKET = '90+'",
    ),
    (
        "Break down denied claims by aging bucket.",
        "SELECT AR_AGING_BUCKET, COUNT(*) AS claim_count FROM Fact_Claims_Adjudication WHERE ADJUDICATION_STATUS = 'Denied' GROUP BY AR_AGING_BUCKET ORDER BY AR_AGING_BUCKET",
    ),
    (
        "Compare inpatient and outpatient performance.",
        "SELECT CLAIM_TYPE, COUNT(*) AS total_claims, COUNT(*) FILTER (WHERE ADJUDICATION_STATUS = 'Denied') AS denied_claims, ROUND(COUNT(*) FILTER (WHERE ADJUDICATION_STATUS = 'Paid - First Pass') * 100.0 / COUNT(*), 2) AS clean_claim_rate_pct, ROUND(SUM(BILLED_PAID_AMT), 2) AS total_paid_amt FROM Fact_Claims_Adjudication GROUP BY CLAIM_TYPE",
    ),
    (
        "What is our net collection ratio?",
        "WITH avg_paid AS (SELECT CLAIM_TYPE, AVG(BILLED_PAID_AMT) AS avg_paid_amt FROM Fact_Claims_Adjudication WHERE ADJUDICATION_STATUS = 'Paid - First Pass' GROUP BY CLAIM_TYPE), enriched AS (SELECT f.BILLED_PAID_AMT, CASE WHEN f.ADJUDICATION_STATUS = 'Denied' THEN a.avg_paid_amt ELSE f.BILLED_PAID_AMT END AS expected_amt FROM Fact_Claims_Adjudication f JOIN avg_paid a ON f.CLAIM_TYPE = a.CLAIM_TYPE) SELECT ROUND(SUM(BILLED_PAID_AMT) * 100.0 / NULLIF(SUM(expected_amt), 0), 2) AS net_collection_ratio_pct FROM enriched",
    ),
    (
        "What is total expected revenue?",
        "WITH avg_paid AS (SELECT CLAIM_TYPE, AVG(BILLED_PAID_AMT) AS avg_paid_amt FROM Fact_Claims_Adjudication WHERE ADJUDICATION_STATUS = 'Paid - First Pass' GROUP BY CLAIM_TYPE) SELECT ROUND(SUM(CASE WHEN f.ADJUDICATION_STATUS = 'Denied' THEN a.avg_paid_amt ELSE f.BILLED_PAID_AMT END), 2) AS total_expected_revenue FROM Fact_Claims_Adjudication f JOIN avg_paid a ON f.CLAIM_TYPE = a.CLAIM_TYPE",
    ),
    (
        "Which denial reasons are front-end problems?",
        "SELECT f.CARC_CODE, c.DESCRIPTION, COUNT(*) AS denial_count FROM Fact_Claims_Adjudication f JOIN Dim_CARC_Denials c ON f.CARC_CODE = c.CARC_CODE WHERE f.ADJUDICATION_STATUS = 'Denied' AND c.PREVENTABILITY_BUCKET = 'Preventable - Front-End' GROUP BY f.CARC_CODE, c.DESCRIPTION ORDER BY denial_count DESC",
    ),
    (
        "How many claims were denied for missing prior authorization?",
        "SELECT COUNT(*) AS denial_count FROM Fact_Claims_Adjudication WHERE ADJUDICATION_STATUS = 'Denied' AND CARC_CODE = '197'",
    ),
    (
        "How many duplicate claim denials do we have?",
        "SELECT COUNT(*) AS denial_count FROM Fact_Claims_Adjudication WHERE ADJUDICATION_STATUS = 'Denied' AND CARC_CODE = '18'",
    ),
    (
        "Show denial counts by CARC code with the preventability bucket.",
        "SELECT f.CARC_CODE, c.DESCRIPTION, c.PREVENTABILITY_BUCKET, COUNT(*) AS denial_count FROM Fact_Claims_Adjudication f JOIN Dim_CARC_Denials c ON f.CARC_CODE = c.CARC_CODE WHERE f.ADJUDICATION_STATUS = 'Denied' GROUP BY f.CARC_CODE, c.DESCRIPTION, c.PREVENTABILITY_BUCKET ORDER BY denial_count DESC",
    ),
    (
        "Which providers have the most preventable denials?",
        "SELECT f.PROVIDER_ID, COUNT(*) AS preventable_denials FROM Fact_Claims_Adjudication f JOIN Dim_CARC_Denials c ON f.CARC_CODE = c.CARC_CODE WHERE f.ADJUDICATION_STATUS = 'Denied' AND c.PREVENTABILITY_BUCKET LIKE 'Preventable%' GROUP BY f.PROVIDER_ID ORDER BY preventable_denials DESC LIMIT 10",
    ),
    (
        "What is the average paid amount for a clean inpatient claim?",
        "SELECT ROUND(AVG(BILLED_PAID_AMT), 2) AS avg_paid_amt FROM Fact_Claims_Adjudication WHERE ADJUDICATION_STATUS = 'Paid - First Pass' AND CLAIM_TYPE = 'Inpatient'",
    ),
    (
        "How many claims do we have in total?",
        "SELECT COUNT(*) AS total_claims FROM Fact_Claims_Adjudication",
    ),
    (
        "How many distinct providers submitted claims?",
        "SELECT COUNT(DISTINCT PROVIDER_ID) AS provider_count FROM Fact_Claims_Adjudication",
    ),
    (
        "Show denial rate by patient chronic condition count.",
        "SELECT p.CHRONIC_CONDITION_COUNT, COUNT(*) AS total_claims, ROUND(COUNT(*) FILTER (WHERE f.ADJUDICATION_STATUS = 'Denied') * 100.0 / COUNT(*), 2) AS denial_rate_pct FROM Fact_Claims_Adjudication f JOIN Dim_Patient p ON f.PATIENT_ID = p.PATIENT_ID GROUP BY p.CHRONIC_CONDITION_COUNT ORDER BY p.CHRONIC_CONDITION_COUNT",
    ),
    (
        "What is the denial rate by state?",
        "SELECT p.SP_STATE_CODE, COUNT(*) AS total_claims, ROUND(COUNT(*) FILTER (WHERE f.ADJUDICATION_STATUS = 'Denied') * 100.0 / COUNT(*), 2) AS denial_rate_pct FROM Fact_Claims_Adjudication f JOIN Dim_Patient p ON f.PATIENT_ID = p.PATIENT_ID GROUP BY p.SP_STATE_CODE ORDER BY denial_rate_pct DESC",
    ),
    (
        "Show the most common diagnoses on denied claims.",
        "SELECT d.DIAGNOSIS_CODE, d.DESCRIPTION, COUNT(*) AS denial_count FROM Fact_Claims_Adjudication f JOIN Dim_Diagnosis d ON f.DIAGNOSIS_CODE = d.DIAGNOSIS_CODE WHERE f.ADJUDICATION_STATUS = 'Denied' GROUP BY d.DIAGNOSIS_CODE, d.DESCRIPTION ORDER BY denial_count DESC LIMIT 10",
    ),
    (
        "Which diagnosis categories have the highest denial rate?",
        "SELECT d.DIAGNOSIS_CATEGORY_APPROX, COUNT(*) AS total_claims, ROUND(COUNT(*) FILTER (WHERE f.ADJUDICATION_STATUS = 'Denied') * 100.0 / COUNT(*), 2) AS denial_rate_pct FROM Fact_Claims_Adjudication f JOIN Dim_Diagnosis d ON f.DIAGNOSIS_CODE = d.DIAGNOSIS_CODE GROUP BY d.DIAGNOSIS_CATEGORY_APPROX ORDER BY denial_rate_pct DESC",
    ),
    (
        "How many claims were submitted each year?",
        "SELECT YEAR(CLM_FROM_DT) AS claim_year, COUNT(*) AS total_claims FROM Fact_Claims_Adjudication GROUP BY claim_year ORDER BY claim_year",
    ),
    (
        "Show monthly denial trend for 2009.",
        "SELECT MONTH(CLM_FROM_DT) AS claim_month, COUNT(*) FILTER (WHERE ADJUDICATION_STATUS = 'Denied') AS denied_claims, COUNT(*) AS total_claims FROM Fact_Claims_Adjudication WHERE YEAR(CLM_FROM_DT) = 2009 GROUP BY claim_month ORDER BY claim_month",
    ),
    (
        "What is the average age of patients with denied claims?",
        "SELECT ROUND(AVG(p.AGE_APPROX), 1) AS avg_age FROM Fact_Claims_Adjudication f JOIN Dim_Patient p ON f.PATIENT_ID = p.PATIENT_ID WHERE f.ADJUDICATION_STATUS = 'Denied'",
    ),
    (
        "List the highest volume providers.",
        "SELECT PROVIDER_ID, COUNT(*) AS total_claims FROM Fact_Claims_Adjudication GROUP BY PROVIDER_ID ORDER BY total_claims DESC LIMIT 10",
    ),
    (
        "How many claims have no denial reason recorded?",
        "SELECT COUNT(*) AS claim_count FROM Fact_Claims_Adjudication WHERE CARC_CODE IS NULL",
    ),
    (
        "Are there any denied claims missing a CARC code?",
        "SELECT COUNT(*) AS claim_count FROM Fact_Claims_Adjudication WHERE ADJUDICATION_STATUS = 'Denied' AND CARC_CODE IS NULL",
    ),
    (
        "Find claims whose diagnosis code is not in the diagnosis dimension.",
        "SELECT f.CLAIM_ID FROM Fact_Claims_Adjudication f LEFT JOIN Dim_Diagnosis d ON f.DIAGNOSIS_CODE = d.DIAGNOSIS_CODE WHERE f.DIAGNOSIS_CODE IS NOT NULL AND d.DIAGNOSIS_CODE IS NULL",
    ),
    (
        "Show average days since submission for denied claims by provider.",
        "SELECT PROVIDER_ID, ROUND(AVG(DAYS_SINCE_SUBMISSION), 1) AS avg_days_outstanding, COUNT(*) AS denied_claims FROM Fact_Claims_Adjudication WHERE ADJUDICATION_STATUS = 'Denied' GROUP BY PROVIDER_ID HAVING COUNT(*) >= 20 ORDER BY avg_days_outstanding DESC LIMIT 10",
    ),
    (
        "What share of total paid dollars comes from inpatient claims?",
        "SELECT ROUND(SUM(BILLED_PAID_AMT) FILTER (WHERE CLAIM_TYPE = 'Inpatient') * 100.0 / SUM(BILLED_PAID_AMT), 2) AS inpatient_share_pct FROM Fact_Claims_Adjudication",
    ),
    (
        "Which patients have the most denied claims?",
        "SELECT PATIENT_ID, COUNT(*) AS denied_claims FROM Fact_Claims_Adjudication WHERE ADJUDICATION_STATUS = 'Denied' GROUP BY PATIENT_ID ORDER BY denied_claims DESC LIMIT 10",
    ),
    (
        "Show denial rate for patients who have died versus those who have not.",
        "SELECT CASE WHEN p.DEATH_DT IS NULL THEN 'Living' ELSE 'Deceased' END AS status, COUNT(*) AS total_claims, ROUND(COUNT(*) FILTER (WHERE f.ADJUDICATION_STATUS = 'Denied') * 100.0 / COUNT(*), 2) AS denial_rate_pct FROM Fact_Claims_Adjudication f JOIN Dim_Patient p ON f.PATIENT_ID = p.PATIENT_ID GROUP BY status",
    ),
    (
        "List CARC codes that never appear on a denied claim.",
        "SELECT c.CARC_CODE, c.DESCRIPTION FROM Dim_CARC_Denials c LEFT JOIN Fact_Claims_Adjudication f ON c.CARC_CODE = f.CARC_CODE WHERE f.CARC_CODE IS NULL",
    ),
]

# Eval-only questions. Three kinds, so the eval measures generalisation rather
# than recall: paraphrases of trained questions, compositions of trained
# concepts, and query shapes that appear nowhere in training.
EVAL_PAIRS: list[tuple[str, str, str]] = [
    # (question, sql, category)
    (
        "What percentage of our claims pay on the first pass?",
        "SELECT ROUND(COUNT(*) FILTER (WHERE ADJUDICATION_STATUS = 'Paid - First Pass') * 100.0 / COUNT(*), 2) AS clean_claim_rate_pct FROM Fact_Claims_Adjudication",
        "paraphrase",
    ),
    (
        "How often do claims get rejected?",
        "SELECT ROUND(COUNT(*) FILTER (WHERE ADJUDICATION_STATUS = 'Denied') * 100.0 / COUNT(*), 2) AS denial_rate_pct FROM Fact_Claims_Adjudication",
        "paraphrase",
    ),
    (
        "Give me the ten worst performing providers on first pass yield.",
        "SELECT PROVIDER_ID, COUNT(*) AS total_claims, ROUND(COUNT(*) FILTER (WHERE ADJUDICATION_STATUS = 'Paid - First Pass') * 100.0 / COUNT(*), 2) AS clean_claim_rate_pct FROM Fact_Claims_Adjudication GROUP BY PROVIDER_ID HAVING COUNT(*) >= 20 ORDER BY clean_claim_rate_pct ASC LIMIT 10",
        "paraphrase",
    ),
    (
        "Which single denial reason costs us the most money?",
        "SELECT f.CARC_CODE, c.DESCRIPTION, COUNT(*) AS denial_count, ROUND(COUNT(*) * (SELECT AVG(BILLED_PAID_AMT) FROM Fact_Claims_Adjudication WHERE ADJUDICATION_STATUS = 'Paid - First Pass'), 2) AS estimated_financial_loss FROM Fact_Claims_Adjudication f JOIN Dim_CARC_Denials c ON f.CARC_CODE = c.CARC_CODE WHERE f.ADJUDICATION_STATUS = 'Denied' GROUP BY f.CARC_CODE, c.DESCRIPTION ORDER BY estimated_financial_loss DESC LIMIT 1",
        "paraphrase",
    ),
    (
        "How much money is tied up in claims older than ninety days?",
        "WITH avg_paid AS (SELECT CLAIM_TYPE, AVG(BILLED_PAID_AMT) AS avg_paid_amt FROM Fact_Claims_Adjudication WHERE ADJUDICATION_STATUS = 'Paid - First Pass' GROUP BY CLAIM_TYPE) SELECT ROUND(SUM(a.avg_paid_amt), 2) AS at_risk_amt_proxy, COUNT(*) AS claim_count FROM Fact_Claims_Adjudication f JOIN avg_paid a ON f.CLAIM_TYPE = a.CLAIM_TYPE WHERE f.ADJUDICATION_STATUS = 'Denied' AND f.AR_AGING_BUCKET = '90+'",
        "paraphrase",
    ),
    (
        "What fraction of denials could we have avoided?",
        "SELECT ROUND(COUNT(*) FILTER (WHERE c.PREVENTABILITY_BUCKET LIKE 'Preventable%') * 100.0 / COUNT(*), 2) AS preventable_pct FROM Fact_Claims_Adjudication f JOIN Dim_CARC_Denials c ON f.CARC_CODE = c.CARC_CODE WHERE f.ADJUDICATION_STATUS = 'Denied'",
        "paraphrase",
    ),
    (
        "Show clean claim rate by claim type for 2010 only.",
        "SELECT CLAIM_TYPE, ROUND(COUNT(*) FILTER (WHERE ADJUDICATION_STATUS = 'Paid - First Pass') * 100.0 / COUNT(*), 2) AS clean_claim_rate_pct FROM Fact_Claims_Adjudication WHERE YEAR(CLM_FROM_DT) = 2010 GROUP BY CLAIM_TYPE",
        "composition",
    ),
    (
        "For inpatient claims only, which providers have the worst clean claim rate?",
        "SELECT PROVIDER_ID, COUNT(*) AS total_claims, ROUND(COUNT(*) FILTER (WHERE ADJUDICATION_STATUS = 'Paid - First Pass') * 100.0 / COUNT(*), 2) AS clean_claim_rate_pct FROM Fact_Claims_Adjudication WHERE CLAIM_TYPE = 'Inpatient' GROUP BY PROVIDER_ID HAVING COUNT(*) >= 20 ORDER BY clean_claim_rate_pct ASC LIMIT 10",
        "composition",
    ),
    (
        "Show preventable denial counts broken down by claim type.",
        "SELECT f.CLAIM_TYPE, COUNT(*) AS preventable_denials FROM Fact_Claims_Adjudication f JOIN Dim_CARC_Denials c ON f.CARC_CODE = c.CARC_CODE WHERE f.ADJUDICATION_STATUS = 'Denied' AND c.PREVENTABILITY_BUCKET LIKE 'Preventable%' GROUP BY f.CLAIM_TYPE",
        "composition",
    ),
    (
        "What is the aged A/R exposure for outpatient claims?",
        "WITH avg_paid AS (SELECT CLAIM_TYPE, AVG(BILLED_PAID_AMT) AS avg_paid_amt FROM Fact_Claims_Adjudication WHERE ADJUDICATION_STATUS = 'Paid - First Pass' GROUP BY CLAIM_TYPE) SELECT ROUND(SUM(a.avg_paid_amt), 2) AS at_risk_amt_proxy FROM Fact_Claims_Adjudication f JOIN avg_paid a ON f.CLAIM_TYPE = a.CLAIM_TYPE WHERE f.ADJUDICATION_STATUS = 'Denied' AND f.AR_AGING_BUCKET = '90+' AND f.CLAIM_TYPE = 'Outpatient'",
        "composition",
    ),
    (
        "Which diagnosis category has the most preventable denials?",
        "SELECT d.DIAGNOSIS_CATEGORY_APPROX, COUNT(*) AS preventable_denials FROM Fact_Claims_Adjudication f JOIN Dim_CARC_Denials c ON f.CARC_CODE = c.CARC_CODE JOIN Dim_Diagnosis d ON f.DIAGNOSIS_CODE = d.DIAGNOSIS_CODE WHERE f.ADJUDICATION_STATUS = 'Denied' AND c.PREVENTABILITY_BUCKET LIKE 'Preventable%' GROUP BY d.DIAGNOSIS_CATEGORY_APPROX ORDER BY preventable_denials DESC LIMIT 1",
        "composition",
    ),
    (
        "Compare the clean claim rate of the top ten providers by volume against everyone else.",
        "WITH top_providers AS (SELECT PROVIDER_ID FROM Fact_Claims_Adjudication GROUP BY PROVIDER_ID ORDER BY COUNT(*) DESC LIMIT 10) SELECT CASE WHEN f.PROVIDER_ID IN (SELECT PROVIDER_ID FROM top_providers) THEN 'Top 10 by volume' ELSE 'All others' END AS cohort, COUNT(*) AS total_claims, ROUND(COUNT(*) FILTER (WHERE f.ADJUDICATION_STATUS = 'Paid - First Pass') * 100.0 / COUNT(*), 2) AS clean_claim_rate_pct FROM Fact_Claims_Adjudication f GROUP BY cohort",
        "novel_shape",
    ),
    (
        "Rank providers by preventable denial rate and show their position.",
        "SELECT f.PROVIDER_ID, COUNT(*) FILTER (WHERE c.PREVENTABILITY_BUCKET LIKE 'Preventable%') AS preventable_denials, ROUND(COUNT(*) FILTER (WHERE c.PREVENTABILITY_BUCKET LIKE 'Preventable%') * 100.0 / COUNT(*), 2) AS preventable_rate_pct, RANK() OVER (ORDER BY COUNT(*) FILTER (WHERE c.PREVENTABILITY_BUCKET LIKE 'Preventable%') * 100.0 / COUNT(*) DESC) AS rank_position FROM Fact_Claims_Adjudication f JOIN Dim_CARC_Denials c ON f.CARC_CODE = c.CARC_CODE WHERE f.ADJUDICATION_STATUS = 'Denied' GROUP BY f.PROVIDER_ID HAVING COUNT(*) >= 20 ORDER BY rank_position LIMIT 10",
        "novel_shape",
    ),
    (
        "Show the running total of denied claims by month across the whole period.",
        "SELECT DATE_TRUNC('month', CLM_FROM_DT) AS claim_month, COUNT(*) AS denied_claims, SUM(COUNT(*)) OVER (ORDER BY DATE_TRUNC('month', CLM_FROM_DT)) AS running_total FROM Fact_Claims_Adjudication WHERE ADJUDICATION_STATUS = 'Denied' GROUP BY claim_month ORDER BY claim_month",
        "novel_shape",
    ),
    (
        "Which providers are in the bottom quartile for clean claim rate?",
        "WITH rates AS (SELECT PROVIDER_ID, COUNT(*) AS total_claims, COUNT(*) FILTER (WHERE ADJUDICATION_STATUS = 'Paid - First Pass') * 100.0 / COUNT(*) AS clean_claim_rate_pct FROM Fact_Claims_Adjudication GROUP BY PROVIDER_ID HAVING COUNT(*) >= 20) SELECT PROVIDER_ID, total_claims, ROUND(clean_claim_rate_pct, 2) AS clean_claim_rate_pct FROM rates WHERE clean_claim_rate_pct <= (SELECT QUANTILE_CONT(clean_claim_rate_pct, 0.25) FROM rates) ORDER BY clean_claim_rate_pct",
        "novel_shape",
    ),
    (
        "For each aging bucket, what is the single most common denial reason?",
        "WITH ranked AS (SELECT f.AR_AGING_BUCKET, f.CARC_CODE, c.DESCRIPTION, COUNT(*) AS denial_count, ROW_NUMBER() OVER (PARTITION BY f.AR_AGING_BUCKET ORDER BY COUNT(*) DESC) AS rn FROM Fact_Claims_Adjudication f JOIN Dim_CARC_Denials c ON f.CARC_CODE = c.CARC_CODE WHERE f.ADJUDICATION_STATUS = 'Denied' GROUP BY f.AR_AGING_BUCKET, f.CARC_CODE, c.DESCRIPTION) SELECT AR_AGING_BUCKET, CARC_CODE, DESCRIPTION, denial_count FROM ranked WHERE rn = 1 ORDER BY AR_AGING_BUCKET",
        "novel_shape",
    ),
    (
        "What is the median approximate patient age on denied inpatient claims?",
        "SELECT MEDIAN(p.AGE_APPROX) AS median_age FROM Fact_Claims_Adjudication f JOIN Dim_Patient p ON f.PATIENT_ID = p.PATIENT_ID WHERE f.ADJUDICATION_STATUS = 'Denied' AND f.CLAIM_TYPE = 'Inpatient'",
        "novel_shape",
    ),
]


# ---------------------------------------------------------------------------
# Tier 2 — templated coverage of the value space
# ---------------------------------------------------------------------------

def _templated_pairs(target: int, seed: int = 42) -> list[tuple[str, str]]:
    """
    Enumerate the value space rather than sampling it.

    Sampling templates at random collides heavily — an earlier version asked for
    170 pairs and produced 46 unique ones, because a handful of templates times a
    handful of values is a small set. Enumerating each axis explicitly gives full
    coverage of the values a model has to learn to quote correctly (CARC codes,
    status strings, bucket labels) with no duplicates to dedupe away.
    """
    pairs: list[tuple[str, str]] = []
    years = [2008, 2009, 2010]

    # One pair per CARC code, twice over: volume and lookup.
    for code, _desc, _bucket in CARC_CODES:
        pairs.append((
            f"How many claims were denied with CARC code {code}?",
            f"SELECT COUNT(*) AS denial_count FROM Fact_Claims_Adjudication WHERE ADJUDICATION_STATUS = 'Denied' AND CARC_CODE = '{code}'",
        ))
        pairs.append((
            f"What does CARC code {code} mean?",
            f"SELECT CARC_CODE, DESCRIPTION, PREVENTABILITY_BUCKET FROM Dim_CARC_Denials WHERE CARC_CODE = '{code}'",
        ))

    for claim_type in CLAIM_TYPES:
        low = claim_type.lower()
        pairs += [
            (
                f"How many {low} claims do we have?",
                f"SELECT COUNT(*) AS total_claims FROM Fact_Claims_Adjudication WHERE CLAIM_TYPE = '{claim_type}'",
            ),
            (
                f"What is the clean claim rate for {low} claims?",
                f"SELECT ROUND(COUNT(*) FILTER (WHERE ADJUDICATION_STATUS = 'Paid - First Pass') * 100.0 / COUNT(*), 2) AS clean_claim_rate_pct FROM Fact_Claims_Adjudication WHERE CLAIM_TYPE = '{claim_type}'",
            ),
            (
                f"How many {low} claims were denied?",
                f"SELECT COUNT(*) AS denied_claims FROM Fact_Claims_Adjudication WHERE CLAIM_TYPE = '{claim_type}' AND ADJUDICATION_STATUS = 'Denied'",
            ),
            (
                f"What is the total paid amount for {low} claims?",
                f"SELECT ROUND(SUM(BILLED_PAID_AMT), 2) AS total_paid_amt FROM Fact_Claims_Adjudication WHERE CLAIM_TYPE = '{claim_type}'",
            ),
        ]
        for year in years:
            pairs.append((
                f"What was the total paid amount for {low} claims in {year}?",
                f"SELECT ROUND(SUM(BILLED_PAID_AMT), 2) AS total_paid_amt FROM Fact_Claims_Adjudication WHERE CLAIM_TYPE = '{claim_type}' AND YEAR(CLM_FROM_DT) = {year}",
            ))
            pairs.append((
                f"How many {low} claims were denied in {year}?",
                f"SELECT COUNT(*) AS denied_claims FROM Fact_Claims_Adjudication WHERE CLAIM_TYPE = '{claim_type}' AND ADJUDICATION_STATUS = 'Denied' AND YEAR(CLM_FROM_DT) = {year}",
            ))

    for bucket in AR_AGING_BUCKETS:
        pairs.append((
            f"How many denied claims are in the {bucket} aging bucket?",
            f"SELECT COUNT(*) AS claim_count FROM Fact_Claims_Adjudication WHERE ADJUDICATION_STATUS = 'Denied' AND AR_AGING_BUCKET = '{bucket}'",
        ))
        pairs.append((
            f"List providers with denied claims in the {bucket} bucket.",
            f"SELECT PROVIDER_ID, COUNT(*) AS claim_count FROM Fact_Claims_Adjudication WHERE ADJUDICATION_STATUS = 'Denied' AND AR_AGING_BUCKET = '{bucket}' GROUP BY PROVIDER_ID ORDER BY claim_count DESC LIMIT 10",
        ))

    for preventability in PREVENTABILITY_BUCKETS:
        pairs.append((
            f"How many denials fall in the {preventability} category?",
            f"SELECT COUNT(*) AS denial_count FROM Fact_Claims_Adjudication f JOIN Dim_CARC_Denials c ON f.CARC_CODE = c.CARC_CODE WHERE f.ADJUDICATION_STATUS = 'Denied' AND c.PREVENTABILITY_BUCKET = '{preventability}'",
        ))
        pairs.append((
            f"Which CARC codes are classified as {preventability}?",
            f"SELECT CARC_CODE, DESCRIPTION FROM Dim_CARC_Denials WHERE PREVENTABILITY_BUCKET = '{preventability}'",
        ))

    for year in years:
        pairs += [
            (
                f"How many claims were submitted in {year}?",
                f"SELECT COUNT(*) AS total_claims FROM Fact_Claims_Adjudication WHERE YEAR(CLM_FROM_DT) = {year}",
            ),
            (
                f"What was the denial rate in {year}?",
                f"SELECT ROUND(COUNT(*) FILTER (WHERE ADJUDICATION_STATUS = 'Denied') * 100.0 / COUNT(*), 2) AS denial_rate_pct FROM Fact_Claims_Adjudication WHERE YEAR(CLM_FROM_DT) = {year}",
            ),
            (
                f"What was the clean claim rate in {year}?",
                f"SELECT ROUND(COUNT(*) FILTER (WHERE ADJUDICATION_STATUS = 'Paid - First Pass') * 100.0 / COUNT(*), 2) AS clean_claim_rate_pct FROM Fact_Claims_Adjudication WHERE YEAR(CLM_FROM_DT) = {year}",
            ),
            (
                f"Show the top denial reasons in {year}.",
                f"SELECT f.CARC_CODE, c.DESCRIPTION, COUNT(*) AS denial_count FROM Fact_Claims_Adjudication f JOIN Dim_CARC_Denials c ON f.CARC_CODE = c.CARC_CODE WHERE f.ADJUDICATION_STATUS = 'Denied' AND YEAR(f.CLM_FROM_DT) = {year} GROUP BY f.CARC_CODE, c.DESCRIPTION ORDER BY denial_count DESC LIMIT 10",
            ),
        ]

    # Deterministic order, then trim to target.
    seen: set[str] = set()
    unique = []
    for q, s in pairs:
        if q not in seen:
            seen.add(q)
            unique.append((q, s))

    random.Random(seed).shuffle(unique)
    return unique[:target]


# ---------------------------------------------------------------------------
# Validation and output
# ---------------------------------------------------------------------------

def validate(pairs: list[tuple[str, str]], label: str) -> list[tuple[str, str]]:
    """Execute every query; drop and report any that fail."""
    good, bad = [], []
    for question, sql in pairs:
        try:
            execute_query(sql)
            good.append((question, sql))
        except Exception as exc:  # noqa: BLE001 — we want the message, not the type
            bad.append((question, str(exc).splitlines()[0]))

    if bad:
        print(f"\n{len(bad)} {label} pair(s) failed to execute and were dropped:")
        for question, err in bad:
            print(f"  - {question}\n      {err}")
    return good


def prepare(n_templated: int = 170, seed: int = 42) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    build_warehouse()

    train_pairs = BUSINESS_PAIRS + _templated_pairs(n_templated, seed=seed)
    train_pairs = validate(train_pairs, "training")

    eval_raw = [(q, s) for q, s, _ in EVAL_PAIRS]
    eval_ok = {q for q, _ in validate(eval_raw, "eval")}
    eval_pairs = [(q, s, c) for q, s, c in EVAL_PAIRS if q in eval_ok]

    # A question must not appear in both splits.
    eval_questions = {q for q, _, _ in eval_pairs}
    leaked = [q for q, _ in train_pairs if q in eval_questions]
    if leaked:
        raise AssertionError(f"Eval questions leaked into training: {leaked}")

    random.Random(seed).shuffle(train_pairs)

    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        for question, sql in train_pairs:
            f.write(json.dumps({
                "question": question,
                "sql": sql,
                "text": build_training_text(question, sql),
            }, ensure_ascii=False) + "\n")

    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        for question, sql, category in eval_pairs:
            f.write(json.dumps({
                "question": question,
                "sql": sql,
                "category": category,
            }, ensure_ascii=False) + "\n")

    n_business = len([p for p in train_pairs if p in BUSINESS_PAIRS])
    print(f"\nTrain: {len(train_pairs)} pairs ({n_business} hand-written, "
          f"{len(train_pairs) - n_business} templated) -> {TRAIN_PATH}")
    print(f"Eval:  {len(eval_pairs)} held-out questions -> {EVAL_PATH}")
    for category in ("paraphrase", "composition", "novel_shape"):
        n = len([p for p in eval_pairs if p[2] == category])
        print(f"         {category:<14} {n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--templated", type=int, default=170)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(n_templated=args.templated, seed=args.seed)
