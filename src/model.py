"""
Model pipeline construction, shared by 04_modeling, 05_evaluation,
06_interpretability, 07_fairness_audit, train_and_save_models.py, and app.py.

Keeping this in one place means the production model definition can
never silently drift between the notebooks and the deployed app.
"""

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_ORDER = ["Detractor", "Passive", "Promoter"]


def split_column_types(X):
    cat_cols = X.select_dtypes(include="str").columns.tolist()
    num_cols = X.select_dtypes(exclude="str").columns.tolist()
    return cat_cols, num_cols


def build_pipeline(cat_cols, num_cols) -> Pipeline:
    """Return the untrained production pipeline: one-hot + scaling +
    class-weighted logistic regression. See 05_evaluation.md for why
    this model (rather than gradient boosting or the ordinal model)
    was selected for production.
    """
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])
    return Pipeline([
        ("prep", prep),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def train_production_model(X_train, y_train) -> Pipeline:
    cat_cols, num_cols = split_column_types(X_train)
    pipe = build_pipeline(cat_cols, num_cols)
    pipe.fit(X_train, y_train)
    return pipe