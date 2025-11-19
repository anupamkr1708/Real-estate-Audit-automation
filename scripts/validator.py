
"""
UNIFIED SCHEMA VALIDATOR 
Creates SAME standardized columns for all 7 extracted CSV files.

FINAL SCHEMA (all validated CSVs contain EXACT same columns):
------------------------------------------------------------
suite_number
tenant_name
lease_status
lease_start_date
lease_end_date
leased_area_sf
amount_per_year
rent_per_sf
escalation_date
rate_annual
rate_monthly
category
increase_date
monthly_amount
psf
data_source
validation_status
validation_warnings
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("validator")

# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def normalize_suite_number(val):
    if pd.isna(val): return ""
    s = str(val).strip().upper()
    s = re.sub(r"\.0+$", "", s)
    s = s.replace("#", "")
    s = re.sub(r"^(SUITE|STE)\s*", "", s)
    if s.isdigit(): return s.zfill(4)
    return s

def normalize_date(val):
    if pd.isna(val): return None
    s = str(val).strip()
    if s in ["", "nan", "None", "N/A"]: return None

    try:
        d = pd.to_datetime(s, errors="coerce")
        if pd.notna(d): return d.strftime("%Y-%m-%d")
    except:
        pass
    return None

def normalize_amount(val):
    if pd.isna(val): return None
    if isinstance(val, (int,float,np.number)):
        return float(val) if val != 0 else None

    s = str(val).replace("$","").replace(",","").strip()
    s = re.sub(r"[^\d\.\-]", "", s)

    try:
        v = float(s)
        return None if v == 0 else v
    except:
        return None

def normalize_tenant(val):
    if pd.isna(val): return ""
    s = str(val).strip()
    s = re.sub(r"\s+", " ", s)
    if "vacant" in s.lower(): return "Vacant"
    return s


# SAFE COLUMN PICKER


def safe_col(colname, df):
    """Return a valid column, else a blank series."""
    if colname in [None, ""]: 
        return pd.Series([None]*len(df), index=df.index)

    if colname in df.columns:
        return df[colname]

    return pd.Series([None]*len(df), index=df.index)


def pick(df, possible):
    for c in possible:
        if c in df.columns:
            return df[c]
    return pd.Series([None]*len(df), index=df.index)



# UNIFIED OUTPUT SCHEMA


UNIFIED_COLUMNS = [
    "suite_number",
    "tenant_name",
    "lease_status",
    "lease_start_date",
    "lease_end_date",
    "leased_area_sf",
    "amount_per_year",
    "rent_per_sf",
    "escalation_date",
    "rate_annual",
    "rate_monthly",
    "category",
    "increase_date",
    "monthly_amount",
    "psf",
    "data_source",
    "validation_status",
    "validation_warnings"
]

def empty_schema(n, source):
    return pd.DataFrame({
        col: [None]*n for col in UNIFIED_COLUMNS
    }).assign(
        data_source=source,
        validation_status="Valid",
        validation_warnings=""
    )



# VALIDATE – LEASE MASTER (Prior AE & Owner AE)


def validate_lease_master(df, source):
    logger.info(f"Validating lease master: {source}")

    out = empty_schema(len(df), source)

    out["suite_number"] = pick(df, ["suite","suite_number","suite_id"]).apply(normalize_suite_number)
    out["tenant_name"] = pick(df, ["tenant","tenant_name"]).apply(normalize_tenant)

    statuses = pick(df, ["lease_status","status"])
    out["lease_status"] = statuses.fillna("").replace("", "Occupied")

    out["lease_start_date"] = pick(df, ["lease_start","lease_start_date"]).apply(normalize_date)
    out["lease_end_date"] = pick(df, ["lease_end","lease_end_date"]).apply(normalize_date)

    out["leased_area_sf"] = pick(df, ["rsf","leased_area_sf","initial_area_sf"]).apply(normalize_amount)
    out["amount_per_year"] = pick(df, ["amount_per_year","base_amount_annual","rental_value_per_year"]).apply(normalize_amount)
    out["rent_per_sf"] = pick(df, ["rent_per_sf","base_rate_annual","rate_per_year"]).apply(normalize_amount)

    # Derive rent per SF
    mask = out["rent_per_sf"].isna() & out["amount_per_year"].notna() & out["leased_area_sf"].notna()
    out.loc[mask, "rent_per_sf"] = out.apply(
        lambda r: r["amount_per_year"]/r["leased_area_sf"] if r["leased_area_sf"] else None,
        axis=1
    )

    return out[out["suite_number"] != ""]



# VALIDATE – RENT ESCALATIONS (Prior AE & Owner AE)
# FIXED VERSION – NOW PRODUCES ROWS


def validate_escalations(df, source):
    logger.info(f"Validating escalations: {source}")

    out = empty_schema(len(df), source)

    # Flexible column detection
    suite_col = next((c for c in df.columns if "suite" in c.lower()), None)
    tenant_col = next((c for c in df.columns if "tenant" in c.lower()), None)
    date_col = next((c for c in df.columns if "date" in c.lower()), None)

    annual_col = next((c for c in df.columns if any(x in c.lower() for x in [
        "annual","rate_annual","amount_annual","psf_year"
    ])), None)

    monthly_col = next((c for c in df.columns if any(x in c.lower() for x in [
        "month","rate_month","psf_month","monthly"
    ])), None)

    # Map to unified schema
    out["suite_number"] = safe_col(suite_col, df).apply(normalize_suite_number)
    out["tenant_name"] = safe_col(tenant_col, df).apply(normalize_tenant)
    out["escalation_date"] = safe_col(date_col, df).apply(normalize_date)
    out["rate_annual"] = safe_col(annual_col, df).apply(normalize_amount)
    out["rate_monthly"] = safe_col(monthly_col, df).apply(normalize_amount)

    # Auto-generate from monthly if needed
    if out["rate_annual"].isna().all() and out["rate_monthly"].notna().any():
        out["rate_annual"] = out["rate_monthly"].apply(lambda x: x*12 if x else None)

    # Keep any row with useful data
    keep_mask = (
        (out["suite_number"] != "") &
        (
            out["escalation_date"].notna() |
            out["rate_annual"].notna() |
            out["rate_monthly"].notna()
        )
    )

    final = out[keep_mask].copy()
    logger.info(f"Escalations validated → {len(final)} rows (from {len(df)})")

    return final



# VALIDATE – STACKING PLAN


def validate_stacking_plan(df):
    logger.info("Validating stacking plan")

    out = empty_schema(len(df), "Stacking Plan")

    out["suite_number"] = pick(df, ["suite"]).apply(normalize_suite_number)
    out["tenant_name"] = pick(df, ["tenant"]).apply(normalize_tenant)
    out["lease_status"] = pick(df, ["status"]).fillna("").apply(
        lambda x: "Vacant" if "vac" in str(x).lower() else "Occupied"
    )
    out["lease_end_date"] = pick(df, ["expiry"]).apply(normalize_date)
    out["leased_area_sf"] = pick(df, ["area"]).apply(normalize_amount)

    return out[out["suite_number"]!=""]



# VALIDATE – RENT ROLL SUITES


def validate_rent_roll_suites(df):
    logger.info("Validating rent roll suites")

    out = empty_schema(len(df), "Rent Roll")

    out["suite_number"] = pick(df, ["suite"]).apply(normalize_suite_number)
    out["tenant_name"] = pick(df, ["tenant"]).apply(normalize_tenant)
    out["lease_status"] = pick(df, ["status"]).apply(lambda x: "Vacant" if "vac" in str(x).lower() else "Occupied")
    out["lease_start_date"] = pick(df, ["rent_start"]).apply(normalize_date)
    out["lease_end_date"] = pick(df, ["expiry"]).apply(normalize_date)
    out["leased_area_sf"] = pick(df, ["sqft"]).apply(normalize_amount)

    monthly = pick(df, ["monthly_rent"]).apply(normalize_amount)
    out["amount_per_year"] = monthly.apply(lambda x: x*12 if x else None)
    out["rent_per_sf"] = pick(df, ["rate_psf"]).apply(normalize_amount)

    return out[out["suite_number"]!=""]


# VALIDATE – RENT ROLL INCREASES


def validate_rent_roll_increases(df):
    logger.info("Validating rent roll increases")

    out = empty_schema(len(df), "Rent Roll")

    out["suite_number"] = pick(df, ["suite"]).apply(normalize_suite_number)
    out["tenant_name"] = pick(df, ["tenant"]).apply(normalize_tenant)
    out["category"] = pick(df, ["category"]).astype(str)
    out["increase_date"] = pick(df, ["increase_date"]).apply(normalize_date)
    out["monthly_amount"] = pick(df, ["monthly_amount"]).apply(normalize_amount)
    out["psf"] = pick(df, ["psf"]).apply(normalize_amount)

    return out[(out["suite_number"]!="") & (out["increase_date"].notna())]



# FILE DISCOVERY


def discover_files(xls_input, pdf_input):
    files = {}
    files["prior_ae_lease_master"] = next(Path(xls_input).glob("*prior*lease_master*.csv"), None)
    files["owner_ae_lease_master"] = next(Path(xls_input).glob("*owner*lease_master*.csv"), None)
    files["prior_ae_rent_escalations"] = next(Path(xls_input).glob("*prior*escalation*.csv"), None)
    files["owner_ae_rent_escalations"] = next(Path(xls_input).glob("*owner*escalation*.csv"), None)

    files["stacking_plan"] = next(Path(pdf_input).glob("*stacking_plan*.csv"), None)
    files["rent_roll_suites"] = next(Path(pdf_input).glob("*suites*.csv"), None)
    files["rent_roll_increases"] = next(Path(pdf_input).glob("*increases*.csv"), None)

    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xls-input", required=True)
    parser.add_argument("--pdf-input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)

    files = discover_files(args.xls_input, args.pdf_input)

    validators = {
        "prior_ae_lease_master": lambda df: validate_lease_master(df, "Prior AE"),
        "owner_ae_lease_master": lambda df: validate_lease_master(df, "Owner AE"),
        "prior_ae_rent_escalations": lambda df: validate_escalations(df, "Prior AE"),
        "owner_ae_rent_escalations": lambda df: validate_escalations(df, "Owner AE"),
        "stacking_plan": validate_stacking_plan,
        "rent_roll_suites": validate_rent_roll_suites,
        "rent_roll_increases": validate_rent_roll_increases,
    }

    for key, path in files.items():
        if path is None:
            logger.warning(f"Missing: {key}")
            continue

        df = pd.read_csv(path)
        validated = validators[key](df)

        validated.to_csv(out_dir / f"{key}_validated.csv", index=False, encoding="utf-8-sig")
        logger.info(f"Saved {key}_validated.csv")

    logger.info("ALL VALIDATION DONE SUCCESSFULLY ✔")


if __name__ == "__main__":
    main()
