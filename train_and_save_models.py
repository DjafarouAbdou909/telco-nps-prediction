"""
Trains and saves the NPS models using the library versions installed
locally (avoids the pickle/joblib cross-environment issues documented
earlier: a model trained on scikit-learn 1.8 failed to unpickle on 1.9).

Usage:
    python train_and_save_models.py

Prerequisite: 03_data_preparation must already have produced
data/processed/telco_nps_train.csv and telco_nps_test.csv.
"""

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
import mord

from src.model import train_production_model, split_column_types, TARGET_ORDER

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

TARGET_MAP_NUM = {c: i for i, c in enumerate(TARGET_ORDER)}


def main():
    train = pd.read_csv(DATA_DIR / "telco_nps_train.csv")
    test = pd.read_csv(DATA_DIR / "telco_nps_test.csv")

    X_train = train.drop(columns=["Customer ID", "NPS_Category"])
    X_test = test.drop(columns=["Customer ID", "NPS_Category"])
    y_train = train["NPS_Category"]
    y_train_ord = y_train.map(TARGET_MAP_NUM)

    cat_cols, num_cols = split_column_types(X_train)

    print("Training Logistic Regression...")
    pipe_lr = train_production_model(X_train, y_train)
    joblib.dump(pipe_lr, MODELS_DIR / "nps_model_logreg.joblib")

    print("Training Gradient Boosting...")
    X_train_gb = X_train.copy()
    for c in cat_cols:
        X_train_gb[c] = X_train_gb[c].astype("category")
    gb = HistGradientBoostingClassifier(
        categorical_features=[X_train_gb.columns.get_loc(c) for c in cat_cols],
        class_weight="balanced", random_state=42
    )
    gb.fit(X_train_gb, y_train)
    joblib.dump(gb, MODELS_DIR / "nps_model_gradientboosting.joblib")

    print("Training Ordinal Regression...")
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    prep_ord = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols)
    ])
    Xtr_o = prep_ord.fit_transform(X_train)
    if hasattr(Xtr_o, "toarray"):
        Xtr_o = Xtr_o.toarray()
    sw = compute_sample_weight("balanced", y_train_ord)
    ord_model = mord.LogisticAT(alpha=1.0)
    ord_model.fit(Xtr_o, y_train_ord, sample_weight=sw)
    joblib.dump({"preprocessor": prep_ord, "model": ord_model},
                MODELS_DIR / "nps_model_ordinal.joblib")

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target_classes_order": TARGET_ORDER,
        "target_mapping_version": "refined (Satisfaction Score 3 tie-broken by Churn Value)",
        "primary_model": "nps_model_logreg.joblib",
        "primary_model_reason": (
            "Best lift on Detractor class (47.6% capture at top 20% outreach) — "
            "aligned with the retention outreach prioritization use case (see 05_evaluation.md)"
        ),
        "feature_columns": X_train.columns.tolist(),
        "categorical_columns": cat_cols,
        "numeric_columns": num_cols,
        "excluded_leakage_columns": [
            "Satisfaction Score", "Churn Score", "Churn Value", "Churn Label",
            "Churn", "Customer Status", "CLTV", "Churn Category", "Churn Reason",
            "Count", "Quarter", "NPS_baseline"
        ],
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "sklearn_version": sklearn.__version__,
        "mord_version": getattr(mord, "__version__", "n/a"),
    }
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Models saved to {MODELS_DIR.resolve()}/")
    print(f"scikit-learn version used: {sklearn.__version__}")


if __name__ == "__main__":
    main()