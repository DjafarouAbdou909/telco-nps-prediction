"""
Data cleaning and feature engineering shared across notebooks, the
training script, and the Gradio app. Extracted from 03_data_preparation
so this logic exists in exactly one place instead of being copy-pasted.
"""

import numpy as np
import pandas as pd

# Columns known to leak into the NPS target, or that carry no
# information (constant across the whole dataset). See
# 03_data_preparation.md for the full justification of each one.
LEAKAGE_AND_NOISE_COLUMNS = [
    "Count", "Quarter", "Satisfaction Score", "Customer Status", "Churn",
    "Churn Label", "Churn Value", "Churn Score", "CLTV", "Churn Category",
    "Churn Reason", "NPS_baseline",
]


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Convert TotalCharges to numeric and fix the 11 blank rows.

    The blanks correspond to customers with tenure == 0 (no first bill
    issued yet) and are replaced with 0.0, not a generic imputed value.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    zero_tenure = df["tenure"] == 0
    df.loc[zero_tenure, "TotalCharges"] = df.loc[zero_tenure, "TotalCharges"].fillna(0.0)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the four engineered features used throughout the project."""
    df = df.copy()

    def count_services(row):
        n = 0
        if row["PhoneService"] == "Yes":
            n += 1
        if row["InternetService"] != "No":
            n += 1
        for c in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                   "TechSupport", "StreamingTV", "StreamingMovies"]:
            if row[c] == "Yes":
                n += 1
        return n

    df["n_services"] = df.apply(count_services, axis=1)
    df["charges_per_service"] = (
        df["MonthlyCharges"] / df["n_services"].replace(0, np.nan)
    ).fillna(df["MonthlyCharges"])
    df["household_size_proxy"] = (df["Partner"] == "Yes").astype(int) + (df["Dependents"] == "Yes").astype(int)
    df["is_autopay"] = df["PaymentMethod"].str.contains("automatic").astype(int)
    return df


def build_modeling_table(df: pd.DataFrame) -> pd.DataFrame:
    """Full 03_data_preparation pipeline: clean, engineer, drop leakage columns.

    Expects df to already contain NPS_Category (built from Satisfaction
    Score elsewhere) and, optionally, NPS_baseline for the sensitivity
    comparison — both are handled correctly either way.
    """
    df = clean_total_charges(df)
    df = engineer_features(df)
    drop_cols = [c for c in LEAKAGE_AND_NOISE_COLUMNS if c in df.columns]
    return df.drop(columns=drop_cols)


def reconstruct_manual_row(feature_cols: list, **raw_inputs) -> pd.DataFrame:
    """Build a single-row DataFrame from raw form inputs (used by app.py),
    computing the same engineered features as build_modeling_table so a
    manually-entered customer is processed identically to a real one.
    """
    phone = raw_inputs["PhoneService"]
    internet = raw_inputs["InternetService"]
    n_services = sum([
        phone == "Yes", internet != "No",
        raw_inputs["OnlineSecurity"] == "Yes",
        raw_inputs["OnlineBackup"] == "Yes",
        raw_inputs["DeviceProtection"] == "Yes",
        raw_inputs["TechSupport"] == "Yes",
        raw_inputs["StreamingTV"] == "Yes",
        raw_inputs["StreamingMovies"] == "Yes",
    ])
    monthly = raw_inputs["MonthlyCharges"]
    charges_per_service = monthly / n_services if n_services > 0 else monthly
    household_size_proxy = int(raw_inputs["Partner"] == "Yes") + int(raw_inputs["Dependents"] == "Yes")
    is_autopay = int("automatic" in raw_inputs["PaymentMethod"])

    row = {**raw_inputs,
           "n_services": n_services,
           "charges_per_service": charges_per_service,
           "household_size_proxy": household_size_proxy,
           "is_autopay": is_autopay}
    return pd.DataFrame([row])[feature_cols]