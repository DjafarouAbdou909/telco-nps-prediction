"""
SHAP explainer setup, shared by 06_interpretability and app.py.

Uses a masker with the full training background (max_samples equal
to the training set size) to avoid the reproducibility bug documented
in 06_interpretability: shap.LinearExplainer subsamples its background
to 100 rows by default, which is not seeded and produces different
SHAP values across environments unless disabled explicitly.
"""

import pandas as pd
import shap


def clean_feature_name(name: str) -> str:
    return name.replace("cat__", "").replace("num__", "")


def build_explainer(pipe, X_train):
    """pipe must already be fitted. Returns (explainer, feat_names)."""
    feat_names = pipe.named_steps["prep"].get_feature_names_out()
    X_train_enc = pipe.named_steps["prep"].transform(X_train)
    clf = pipe.named_steps["clf"]
    masker = shap.maskers.Independent(X_train_enc, max_samples=X_train_enc.shape[0])
    explainer = shap.LinearExplainer(clf, masker, feature_names=feat_names)
    return explainer, feat_names


def top_drivers_for_row(pipe, explainer, feat_names, row_df, target_class="Detractor", n=5):
    """Return the top-n |SHAP value| features pushing toward target_class
    for a single-row DataFrame, as a pandas Series indexed by clean
    feature names (one-hot prefixes stripped).
    """
    classes = pipe.named_steps["clf"].classes_
    class_idx = list(classes).index(target_class)
    row_enc = pipe.named_steps["prep"].transform(row_df)
    sv = explainer(row_enc).values[0][:, class_idx]
    return (
        pd.Series(sv, index=[clean_feature_name(f) for f in feat_names])
        .sort_values(key=abs, ascending=False)
        .head(n)
    )