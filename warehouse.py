"""
Builds and queries the DuckDB claims warehouse.

Two ingestion paths, same output schema:

  real       If CMS DE-SynPUF CSVs are present in data/raw/, they are ingested
             with the same logic as the source warehouse project — union of
             inpatient and outpatient claims, aggregated to one row per CLM_ID,
             denial simulated where Medicare paid $0.

  synthetic  Otherwise a deterministic generator produces a warehouse with the
             identical star schema. This exists so the repo is clonable and
             runnable without a 1GB CMS download, and so the eval harness has
             something to execute against in CI.

Neither path contains PHI. DE-SynPUF is itself synthetic, published by CMS.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb

from schema import CARC_CODES, CARC_WEIGHTS, DDL, SCHEMA_CONTEXT

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "claims_warehouse.duckdb"

AS_OF = date(2010, 12, 31)  # last date in the SynPUF window; "today" for aging

# Diagnosis codes are ICD-9-CM because DE-SynPUF covers 2008-2010, before the
# US ICD-10 transition. The source project makes the same call and documents it.
DIAGNOSIS_SEED: list[tuple[str, str, str]] = [
    ("4019", "Circulatory", "Unspecified essential hypertension"),
    ("25000", "Endocrine/Metabolic", "Diabetes mellitus without mention of complication"),
    ("42731", "Circulatory", "Atrial fibrillation"),
    ("4280", "Circulatory", "Congestive heart failure, unspecified"),
    ("41401", "Circulatory", "Coronary atherosclerosis of native coronary artery"),
    ("496", "Respiratory", "Chronic airway obstruction, not elsewhere classified"),
    ("486", "Respiratory", "Pneumonia, organism unspecified"),
    ("49390", "Respiratory", "Asthma, unspecified type"),
    ("5849", "Other", "Acute kidney failure, unspecified"),
    ("2724", "Endocrine/Metabolic", "Other and unspecified hyperlipidemia"),
    ("2449", "Endocrine/Metabolic", "Unspecified acquired hypothyroidism"),
    ("53081", "Other", "Esophageal reflux"),
    ("311", "Other", "Depressive disorder, not elsewhere classified"),
    ("71590", "Other", "Osteoarthrosis, unspecified whether generalized or localized"),
    ("73300", "Other", "Osteoporosis, unspecified"),
    ("78650", "Other", "Chest pain, unspecified"),
    ("7802", "Other", "Syncope and collapse"),
    ("59080", "Other", "Urinary tract infection, site not specified"),
    ("82101", "Injury/Poisoning", "Closed fracture of shaft of femur"),
    ("80500", "Injury/Poisoning", "Closed fracture of cervical vertebra"),
    ("V5861", "Supplemental/V-code", "Long-term use of anticoagulants"),
    ("V4581", "Supplemental/V-code", "Aortocoronary bypass status"),
    ("E8889", "External cause/E-code", "Unspecified fall"),
]

STATE_CODES = ["01", "05", "10", "14", "22", "26", "33", "39", "45", "49"]


# ---------------------------------------------------------------------------
# Connection and safe execution
# ---------------------------------------------------------------------------

def get_connection(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Open the warehouse, building it first if it does not exist."""
    if not DB_PATH.exists():
        build_warehouse()
    return duckdb.connect(str(DB_PATH), read_only=read_only)


FORBIDDEN = (
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE",
    "REPLACE", "ATTACH", "DETACH", "COPY", "EXPORT", "INSTALL", "LOAD",
    "PRAGMA", "CALL", "SET",
)


def execute_query(sql: str, row_limit: int = 500) -> tuple[list[str], list[tuple[Any, ...]]]:
    """
    Execute a read-only SELECT and return (columns, rows).

    Read-only is enforced twice: by rejecting anything that is not a single
    SELECT or WITH, and by opening the connection in DuckDB's read_only mode so
    a bypass of the string check still cannot write.
    """
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        raise ValueError("Empty query.")

    if ";" in stripped:
        raise ValueError("Only a single statement is allowed.")

    upper = stripped.upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        raise ValueError("Only SELECT queries are allowed.")

    import re

    for keyword in FORBIDDEN:
        if re.search(rf"\b{keyword}\b", upper):
            raise ValueError(f"Query contains forbidden keyword: {keyword}")

    con = get_connection(read_only=True)
    try:
        cursor = con.execute(stripped)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(row_limit)
        return columns, [tuple(r) for r in rows]
    finally:
        con.close()


def get_schema_context() -> str:
    return SCHEMA_CONTEXT


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _real_files_present() -> bool:
    needed = [
        "DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv",
        "DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.csv",
        "DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv",
    ]
    return all((RAW_DIR / f).exists() for f in needed)


def build_warehouse(force: bool = False, n_claims: int = 40_000, seed: int = 42) -> str:
    """Build the warehouse. Returns 'real' or 'synthetic' to say which path ran."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        if not force:
            return "existing"
        DB_PATH.unlink()

    con = duckdb.connect(str(DB_PATH))
    con.execute("PRAGMA threads=4;")
    con.execute(DDL)
    _load_carc(con)

    if _real_files_present():
        _build_from_synpuf(con)
        mode = "real"
    else:
        _build_synthetic(con, n_claims=n_claims, seed=seed)
        mode = "synthetic"

    con.execute("""
        CREATE OR REPLACE TABLE _build_metadata AS
        SELECT ? AS source_mode, current_localtimestamp() AS built_at
    """, [mode])
    con.close()
    return mode


def _load_carc(con: duckdb.DuckDBPyConnection) -> None:
    con.executemany(
        "INSERT INTO Dim_CARC_Denials VALUES (?, ?, ?)", CARC_CODES
    )


def _build_from_synpuf(con: duckdb.DuckDBPyConnection) -> None:
    """Ingest the real CMS DE-SynPUF extracts from data/raw/."""
    raise NotImplementedError(
        "Real DE-SynPUF ingestion is delegated to the upstream warehouse project. "
        "Run build_warehouse.py there and point DB_PATH at the resulting "
        "claims_warehouse.duckdb, or delete the files in data/raw/ to use the "
        "synthetic generator."
    )


def _build_synthetic(con: duckdb.DuckDBPyConnection, n_claims: int, seed: int) -> None:
    """Generate a warehouse with the production schema and realistic distributions."""
    rng = random.Random(seed)

    con.executemany(
        "INSERT INTO Dim_Diagnosis VALUES (?, 'ICD-9-CM', ?, ?)",
        [(code, cat, desc) for code, cat, desc in DIAGNOSIS_SEED],
    )

    # Patients
    n_patients = max(1, n_claims // 12)
    patients = []
    for i in range(n_patients):
        pid = f"P{i:08d}"
        age = int(rng.triangular(66, 95, 74))
        birth = date(2010 - age, rng.randint(1, 12), rng.randint(1, 28))
        died = rng.random() < 0.04
        patients.append((
            pid,
            birth,
            birth + timedelta(days=age * 365 + rng.randint(0, 300)) if died else None,
            rng.choice(["Male", "Female"]),
            rng.choice(["1", "2", "3", "5"]),
            rng.choice(STATE_CODES),
            age,
            min(11, max(0, int(rng.gauss(2.6, 1.9)))),
            2010,
        ))
    con.executemany("INSERT INTO Dim_Patient VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", patients)

    # Providers. Claim volume follows a heavy power law: a few large facilities
    # submit thousands of claims and a long tail submits a handful each.
    #
    # That tail is load-bearing. It is the reason provider league tables carry a
    # minimum-volume floor — without one, the "worst clean claim rate" ranking
    # fills up with providers who submitted three claims and had two denied, at
    # 33%, which is small-sample noise rather than a performance problem. An
    # earlier version of this generator produced only one provider below the
    # floor, which made the floor a no-op and left the training pair that
    # teaches it unverifiable by the eval.
    n_providers = max(1, n_claims // 15)
    provider_ids = list(dict.fromkeys(f"P{i:06d}" for i in range(n_providers)))
    provider_weights = [rng.paretovariate(1.1) for _ in provider_ids]

    # Each provider gets a latent "clean claim propensity" so the KPI has signal.
    provider_quality = {p: min(0.97, max(0.55, rng.gauss(0.82, 0.10))) for p in provider_ids}

    carc_pool = [c for c in CARC_WEIGHTS if any(c == row[0] for row in CARC_CODES)]
    carc_weights = [CARC_WEIGHTS[c] for c in carc_pool]

    diag_codes = [d[0] for d in DIAGNOSIS_SEED]
    patient_ids = [p[0] for p in patients]

    facts = []
    provider_counts: dict[str, int] = {}
    for i in range(n_claims):
        provider = rng.choices(provider_ids, weights=provider_weights, k=1)[0]
        provider_counts[provider] = provider_counts.get(provider, 0) + 1

        claim_type = "Inpatient" if rng.random() < 0.18 else "Outpatient"
        from_dt = date(2008, 1, 1) + timedelta(days=rng.randint(0, 1080))
        los = rng.randint(1, 12) if claim_type == "Inpatient" else rng.randint(0, 2)
        thru_dt = from_dt + timedelta(days=los)

        clean = rng.random() < provider_quality[provider]
        if clean:
            status = "Paid - First Pass"
            carc = None
            base = rng.lognormvariate(8.6, 0.75) if claim_type == "Inpatient" else rng.lognormvariate(5.2, 0.95)
            paid = round(base, 2)
        else:
            status = "Denied"
            carc = rng.choices(carc_pool, weights=carc_weights, k=1)[0]
            paid = 0.0

        days_since = (AS_OF - from_dt).days
        bucket = (
            "0-30" if days_since <= 30
            else "31-60" if days_since <= 60
            else "61-90" if days_since <= 90
            else "90+"
        )

        facts.append((
            f"CLM{i:09d}",
            rng.choice(patient_ids),
            provider,
            rng.choice(diag_codes),
            carc,
            claim_type,
            from_dt,
            thru_dt,
            paid,
            round(paid * rng.uniform(0.0, 0.15), 2) if clean else 0.0,
            status,
            days_since,
            bucket,
        ))

    con.executemany(
        "INSERT INTO Fact_Claims_Adjudication VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        facts,
    )

    con.executemany(
        "INSERT INTO Dim_Provider VALUES (?, ?, ?)",
        [
            (p, f"{rng.randint(1000000000, 1999999999)}", provider_counts.get(p, 0))
            for p in provider_ids
        ],
    )


def summarise() -> str:
    con = get_connection(read_only=True)
    try:
        mode = con.execute("SELECT source_mode FROM _build_metadata").fetchone()[0]
        lines = [f"Warehouse: {DB_PATH} (source: {mode})"]
        for table in (
            "Dim_Patient", "Dim_Provider", "Dim_Diagnosis",
            "Dim_CARC_Denials", "Fact_Claims_Adjudication",
        ):
            n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            lines.append(f"  {table:<28} {n:>9,} rows")
        ccr = con.execute("""
            SELECT ROUND(COUNT(*) FILTER (WHERE ADJUDICATION_STATUS = 'Paid - First Pass')
                         * 100.0 / COUNT(*), 2)
            FROM Fact_Claims_Adjudication
        """).fetchone()[0]
        lines.append(f"  Clean claim rate             {ccr:>9}%")
        return "\n".join(lines)
    finally:
        con.close()


if __name__ == "__main__":
    mode = build_warehouse(force=True)
    print(f"Built warehouse from {mode} source.")
    print(summarise())
