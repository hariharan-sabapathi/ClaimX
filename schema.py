"""
Star schema definition for the claims adjudication warehouse.

This mirrors the marts built by Medical-Insurance-Claims-And-Denial-Analytics:
Dim_Patient, Dim_Provider, Dim_Diagnosis, Dim_CARC_Denials, and the
Fact_Claims_Adjudication grain of one row per claim. Column names and types are
kept identical so a model fine-tuned here transfers to that warehouse unchanged.

Everything downstream — dataset generation, the prompt, the eval harness —
reads the schema from this module, so there is exactly one place to edit it.
"""

from __future__ import annotations

DDL = """
CREATE TABLE IF NOT EXISTS Dim_Patient (
    PATIENT_ID               VARCHAR PRIMARY KEY,
    BIRTH_DT                 DATE,
    DEATH_DT                 DATE,
    SEX                      VARCHAR,
    BENE_RACE_CD             VARCHAR,
    SP_STATE_CODE            VARCHAR,
    AGE_APPROX               INTEGER,
    CHRONIC_CONDITION_COUNT  INTEGER,
    SNAPSHOT_YEAR            INTEGER
);

CREATE TABLE IF NOT EXISTS Dim_Provider (
    PROVIDER_ID    VARCHAR PRIMARY KEY,
    ATTENDING_NPI  VARCHAR,
    CLAIM_VOLUME   INTEGER
);

CREATE TABLE IF NOT EXISTS Dim_Diagnosis (
    DIAGNOSIS_CODE             VARCHAR PRIMARY KEY,
    CODE_SYSTEM                VARCHAR,
    DIAGNOSIS_CATEGORY_APPROX  VARCHAR,
    DESCRIPTION                VARCHAR
);

CREATE TABLE IF NOT EXISTS Dim_CARC_Denials (
    CARC_CODE              VARCHAR PRIMARY KEY,
    DESCRIPTION            VARCHAR,
    PREVENTABILITY_BUCKET  VARCHAR
);

CREATE TABLE IF NOT EXISTS Fact_Claims_Adjudication (
    CLAIM_ID               VARCHAR PRIMARY KEY,
    PATIENT_ID             VARCHAR,
    PROVIDER_ID            VARCHAR,
    DIAGNOSIS_CODE         VARCHAR,
    CARC_CODE              VARCHAR,
    CLAIM_TYPE             VARCHAR,
    CLM_FROM_DT            DATE,
    CLM_THRU_DT            DATE,
    BILLED_PAID_AMT        DOUBLE,
    PRIMARY_PYR_PD_AMT     DOUBLE,
    ADJUDICATION_STATUS    VARCHAR,
    DAYS_SINCE_SUBMISSION  INTEGER,
    AR_AGING_BUCKET        VARCHAR
);
"""

# The compact schema string that goes into every prompt. Deliberately terse:
# it is prepended to all 200 training examples and to every inference call, so
# tokens here are paid for on every single request.
SCHEMA_CONTEXT = """Tables:
Dim_Patient(PATIENT_ID, BIRTH_DT, DEATH_DT, SEX, BENE_RACE_CD, SP_STATE_CODE, AGE_APPROX, CHRONIC_CONDITION_COUNT, SNAPSHOT_YEAR)
Dim_Provider(PROVIDER_ID, ATTENDING_NPI, CLAIM_VOLUME)
Dim_Diagnosis(DIAGNOSIS_CODE, CODE_SYSTEM, DIAGNOSIS_CATEGORY_APPROX, DESCRIPTION)
Dim_CARC_Denials(CARC_CODE, DESCRIPTION, PREVENTABILITY_BUCKET)
Fact_Claims_Adjudication(CLAIM_ID, PATIENT_ID, PROVIDER_ID, DIAGNOSIS_CODE, CARC_CODE, CLAIM_TYPE, CLM_FROM_DT, CLM_THRU_DT, BILLED_PAID_AMT, PRIMARY_PYR_PD_AMT, ADJUDICATION_STATUS, DAYS_SINCE_SUBMISSION, AR_AGING_BUCKET)"""

# Enumerated values a model cannot infer from column names alone.
ADJUDICATION_STATUSES = ["Paid - First Pass", "Denied"]
CLAIM_TYPES = ["Inpatient", "Outpatient"]
AR_AGING_BUCKETS = ["0-30", "31-60", "61-90", "90+"]

PREVENTABILITY_BUCKETS = [
    "Preventable - Process",
    "Preventable - Front-End",
    "Non-Preventable - Coverage",
    "Non-Preventable - Clinical",
    "Non-Preventable - Patient Responsibility",
    "Unclassified",
]

# Real X12 Claim Adjustment Reason Codes. Descriptions are abridged from the
# published national code set. The preventability bucket is this project's
# classification, using the same keyword taxonomy as the source warehouse.
CARC_CODES: list[tuple[str, str, str]] = [
    ("1", "Deductible amount", "Non-Preventable - Patient Responsibility"),
    ("2", "Coinsurance amount", "Non-Preventable - Patient Responsibility"),
    ("3", "Co-payment amount", "Non-Preventable - Patient Responsibility"),
    ("4", "The procedure code is inconsistent with the modifier used", "Preventable - Process"),
    ("11", "The diagnosis is inconsistent with the procedure", "Preventable - Process"),
    ("15", "The authorization number is missing, invalid, or does not apply", "Preventable - Front-End"),
    ("16", "Claim/service lacks information or has submission/billing error(s)", "Preventable - Process"),
    ("18", "Exact duplicate claim/service", "Preventable - Process"),
    ("22", "This care may be covered by another payer per coordination of benefits", "Preventable - Process"),
    ("23", "The impact of prior payer(s) adjudication including payments and/or adjustments", "Non-Preventable - Coverage"),
    ("24", "Charges are covered under a capitation agreement/managed care plan", "Non-Preventable - Coverage"),
    ("26", "Expenses incurred prior to coverage", "Preventable - Front-End"),
    ("27", "Expenses incurred after coverage terminated", "Preventable - Front-End"),
    ("29", "The time limit for filing has expired", "Preventable - Process"),
    ("31", "Patient cannot be identified as our insured", "Preventable - Front-End"),
    ("39", "Services denied at the time authorization/pre-certification was requested", "Preventable - Front-End"),
    ("45", "Charge exceeds fee schedule/maximum allowable", "Non-Preventable - Coverage"),
    ("49", "This is a non-covered service because it is a routine/preventive exam", "Non-Preventable - Coverage"),
    ("50", "These are non-covered services because this is not deemed a medical necessity", "Non-Preventable - Clinical"),
    ("54", "Multiple physicians/assistants are not covered in this case", "Non-Preventable - Coverage"),
    ("55", "Procedure/treatment/drug is deemed experimental/investigational", "Non-Preventable - Clinical"),
    ("58", "Treatment was deemed by the payer to have been rendered in an inappropriate setting", "Non-Preventable - Clinical"),
    ("96", "Non-covered charge(s)", "Non-Preventable - Coverage"),
    ("97", "The benefit for this service is included in the payment for another service", "Preventable - Process"),
    ("109", "Claim/service not covered by this payer/contractor", "Non-Preventable - Coverage"),
    ("119", "Benefit maximum for this time period or occurrence has been reached", "Non-Preventable - Coverage"),
    ("140", "Patient/insured health identification number and name do not match", "Preventable - Front-End"),
    ("151", "Payer deems the information submitted does not support this many services", "Non-Preventable - Clinical"),
    ("167", "This/these diagnosis(es) is/are not covered", "Non-Preventable - Coverage"),
    ("181", "Procedure code was invalid on the date of service", "Preventable - Process"),
    ("197", "Precertification/authorization/notification/pre-treatment absent", "Preventable - Front-End"),
    ("204", "This service/equipment/drug is not covered under the patient's current benefit plan", "Non-Preventable - Coverage"),
    ("252", "An attachment/other documentation is required to adjudicate this claim", "Preventable - Process"),
]

# Denial-mix weights mirroring the source warehouse's realistic RCM distribution.
CARC_WEIGHTS: dict[str, int] = {
    "18": 18, "16": 15, "197": 12, "50": 12, "29": 8,
    "27": 8, "96": 8, "119": 6, "109": 6, "1": 4, "2": 3,
    "15": 3, "140": 2, "252": 2, "181": 2, "22": 2, "167": 1,
}

# ---------------------------------------------------------------------------
# Business definitions.
#
# These are the reason this project exists. None of them are recoverable from
# the DDL: a model reading the schema has no way to know that "clean claim
# rate" filters on a specific status string, that provider rankings apply a
# minimum-volume floor, or that a denied claim's dollar value has to be proxied
# because BILLED_PAID_AMT is zero by construction. They live here, in the
# training pairs, and in the fine-tuned weights.
# ---------------------------------------------------------------------------
BUSINESS_RULES = """Definitions:
- A claim is "clean" or "paid first pass" when ADJUDICATION_STATUS = 'Paid - First Pass'.
- Clean claim rate = clean claims * 100.0 / total claims.
- Denial rate = claims with ADJUDICATION_STATUS = 'Denied' * 100.0 / total claims.
- Denied claims have BILLED_PAID_AMT = 0 by construction. Dollar value at risk must be
  proxied by the average BILLED_PAID_AMT of first-pass-paid claims of the same CLAIM_TYPE.
- Provider league tables require HAVING COUNT(*) >= 20 so low-volume providers do not top the ranking.
- Preventable denials are rows in Dim_CARC_Denials where PREVENTABILITY_BUCKET LIKE 'Preventable%'.
- Aged A/R means AR_AGING_BUCKET = '90+'.
"""
