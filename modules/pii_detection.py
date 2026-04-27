"""
InsOps · PII & PHI Detection Module
Detects, flags, and masks sensitive insurance data columns.
Supports HIPAA, GDPR, GLBA compliance signals.
"""
import pandas as pd
import numpy as np
import re
import hashlib


# ── Pattern library ───────────────────────────────────────────────────────────
PII_PATTERNS = {
    "SSN":          r"\b\d{3}-\d{2}-\d{4}\b",
    "Phone":        r"\b(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b",
    "Email":        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "Credit Card":  r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "ZIP Code":     r"\b\d{5}(?:-\d{4})?\b",
    "IP Address":   r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "Date of Birth":r"\b(0?[1-9]|1[0-2])[/-](0?[1-9]|[12]\d|3[01])[/-](\d{2}|\d{4})\b",
}

PHI_PATTERNS = {
    "ICD Code":     r"\b[A-Z]\d{2}\.?\d{0,2}\b",
    "NPI Number":   r"\b\d{10}\b",
    "MRN":          r"\b(MRN|mrn)[-:\s]?\d{6,10}\b",
    "DEA Number":   r"\b[A-Z]{2}\d{7}\b",
}

# Column name keywords that strongly suggest PII/PHI
PII_NAME_KEYWORDS = [
    "name", "first_name", "last_name", "fname", "lname", "fullname",
    "email", "mail", "phone", "mobile", "cell", "tel", "fax",
    "address", "addr", "street", "city", "state", "zip", "postal",
    "ssn", "social", "dob", "birth", "age", "gender", "sex",
    "passport", "license", "driver", "national_id", "taxpayer",
    "ip_address", "device_id", "mac_address", "cookie", "user_id",
    "account", "credit_card", "bank", "routing",
]

PHI_NAME_KEYWORDS = [
    "diagnosis", "icd", "cpt", "procedure", "condition", "disease",
    "medication", "drug", "prescription", "rx", "ndc", "npi",
    "provider", "physician", "doctor", "hospital", "facility",
    "mrn", "patient", "member_id", "claim", "claimant",
    "treatment", "surgery", "lab", "test_result", "vital",
    "disability", "mental", "health_plan", "beneficiary",
]

COMPLIANCE_MAP = {
    "SSN":           ["HIPAA", "GDPR", "GLBA"],
    "Email":         ["GDPR", "GLBA"],
    "Phone":         ["GDPR", "GLBA"],
    "Credit Card":   ["GLBA", "PCI-DSS"],
    "Date of Birth": ["HIPAA", "GDPR"],
    "ICD Code":      ["HIPAA"],
    "NPI Number":    ["HIPAA"],
    "MRN":           ["HIPAA"],
    "name":          ["HIPAA", "GDPR"],
    "address":       ["HIPAA", "GDPR"],
    "diagnosis":     ["HIPAA"],
    "patient":       ["HIPAA"],
    "claim":         ["HIPAA", "GLBA"],
}


def _match_by_name(col: str) -> dict:
    col_l = col.lower()
    matched = {}
    for kw in PII_NAME_KEYWORDS:
        if kw in col_l:
            matched["type"] = "PII"
            matched["reason"] = f"Column name contains '{kw}'"
            matched["pattern_name"] = kw
            return matched
    for kw in PHI_NAME_KEYWORDS:
        if kw in col_l:
            matched["type"] = "PHI"
            matched["reason"] = f"Column name contains '{kw}'"
            matched["pattern_name"] = kw
            return matched
    return {}


def _match_by_content(series: pd.Series) -> dict:
    sample = series.dropna().astype(str).head(200)
    for name, pattern in PII_PATTERNS.items():
        hits = sample.str.contains(pattern, regex=True, na=False).sum()
        if hits > 0:
            return {"type": "PII", "reason": f"Content matches {name} pattern ({hits} samples)", "pattern_name": name}
    for name, pattern in PHI_PATTERNS.items():
        hits = sample.str.contains(pattern, regex=True, na=False).sum()
        if hits > 0:
            return {"type": "PHI", "reason": f"Content matches {name} pattern ({hits} samples)", "pattern_name": name}
    return {}


def detect_pii_phi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scan all columns for PII/PHI by name and content patterns.
    Returns a DataFrame with one row per column.
    """
    results = []
    for col in df.columns:
        row = {"column": col, "type": "Safe", "reason": "No PII/PHI detected",
               "pattern_name": "", "compliance_risk": "", "risk_level": "Low"}

        # Name-based detection first (faster)
        match = _match_by_name(col)
        if not match:
            match = _match_by_content(df[col])

        if match:
            row["type"]           = match["type"]
            row["reason"]         = match["reason"]
            row["pattern_name"]   = match.get("pattern_name", "")
            row["risk_level"]     = "High" if match["type"] == "PHI" else "Medium"

            # Compliance tags
            pk = match.get("pattern_name", "")
            regs = COMPLIANCE_MAP.get(pk, [])
            if match["type"] == "PHI" and "HIPAA" not in regs:
                regs = ["HIPAA"] + regs
            row["compliance_risk"] = ", ".join(regs) if regs else ("HIPAA" if match["type"] == "PHI" else "GDPR")

        results.append(row)

    return pd.DataFrame(results)


def mask_column(series: pd.Series, method: str) -> pd.Series:
    """Apply masking to a Series. Methods: redact, hash, tokenize, partial."""
    # Convert numerical series to string so all methods work uniformly
    series = series.astype(str).replace("nan", np.nan)

    if method == "Redact":
        return pd.Series(["[REDACTED]"] * len(series), index=series.index)

    elif method == "Hash (SHA-256)":
        def _hash(v):
            if pd.isna(v): return v
            return hashlib.sha256(str(v).encode()).hexdigest()[:16]
        return series.apply(_hash)

    elif method == "Tokenize":
        mapping = {v: f"TOKEN_{i:04d}" for i, v in enumerate(series.dropna().unique())}
        return series.map(mapping).fillna("[NULL]")

    elif method == "Partial Mask":
        def _partial(v):
            if pd.isna(v): return v
            s = str(v)
            if len(s) <= 4: return "*" * len(s)
            return s[:2] + "*" * (len(s) - 4) + s[-2:]
        return series.apply(_partial)

    return series


def compute_compliance_score(pii_df: pd.DataFrame, total_cols: int) -> dict:
    """Return per-regulation compliance readiness scores."""
    flagged = pii_df[pii_df["type"] != "Safe"]
    high    = (pii_df["risk_level"] == "High").sum()
    medium  = (pii_df["risk_level"] == "Medium").sum()

    # Base score deductions
    base = 100
    base -= high   * 15
    base -= medium * 8
    base = max(base, 0)

    has_phi = (pii_df["type"] == "PHI").any()
    has_pii = (pii_df["type"] == "PII").any()

    return {
        "overall":  base,
        "HIPAA":    max(0, base - (20 if has_phi else 0)),
        "GDPR":     max(0, base - (15 if has_pii else 0)),
        "GLBA":     max(0, base - (10 if has_pii else 0)),
        "flagged_count": len(flagged),
        "high_risk":     int(high),
        "medium_risk":   int(medium),
    }
