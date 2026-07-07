import json
import sklearn
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
import mord

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

TARGET_ORDER = ["Detractor", "Passive", "Promoter"]
TARGET_MAP_NUM = {c: i for i, c in enumerate(TARGET_ORDER)}


def main():
    train = pd.read_csv(DATA_DIR / "telco_nps_train.csv")
    test = pd.read_csv(DATA_DIR / "telco_nps_test.csv")

    X_train = train.drop(columns=["Customer ID", "NPS_Category"])
    X_test = test.drop(columns=["Customer ID", "NPS_Category"])
    y_train = train["NPS_Category"]
    y_train_ord = y_train.map(TARGET_MAP_NUM)

    cat_cols = X_train.select_dtypes(include="str").columns.tolist()
    num_cols = X_train.select_dtypes(exclude="str").columns.tolist()

    print("Entraînement Logistic Regression...")
    pipe_lr = Pipeline([
        ("prep", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols)
        ])),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    pipe_lr.fit(X_train, y_train)
    joblib.dump(pipe_lr, MODELS_DIR / "nps_model_logreg.joblib")

    print("Entraînement Gradient Boosting...")
    X_train_gb = X_train.copy()
    for c in cat_cols:
        X_train_gb[c] = X_train_gb[c].astype("category")
    gb = HistGradientBoostingClassifier(
        categorical_features=[X_train_gb.columns.get_loc(c) for c in cat_cols],
        class_weight="balanced", random_state=42
    )
    gb.fit(X_train_gb, y_train)
    joblib.dump(gb, MODELS_DIR / "nps_model_gradientboosting.joblib")

    print("Entraînement Ordinal Regression...")
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
            "Count", "Quarter"
        ],
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "sklearn_version": sklearn.__version__,
        "mord_version": getattr(mord, "__version__", "n/a"),
    }
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nTerminé. Modèles sauvegardés dans {MODELS_DIR.resolve()}/")


if __name__ == "__main__":
    main()