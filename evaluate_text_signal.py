"""

Compares three configurations on the same test set:
    1. Tabular only (the production model from 05_evaluation)
    2. Text only (sentence embeddings of the verbatim -> classifier)
    3. Tabular + text combined

Usage:
    python evaluate_text_signal.py

Prerequisite: generate_verbatims.py must have already produced
data/processed/verbatims.csv.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sentence_transformers import SentenceTransformer

from src.model import split_column_types, TARGET_ORDER

TARGET_MAP_NUM = {c: i for i, c in enumerate(TARGET_ORDER)}


def load_data():
    test_full = pd.read_csv("data/processed/telco_nps_test.csv")
    verbatims = pd.read_csv("data/processed/verbatims.csv")
    df = test_full.merge(verbatims[["Customer ID", "verbatim"]], on="Customer ID", how="inner")
    df = df.dropna(subset=["verbatim"])
    print(f"{len(df)} customers with both tabular data and a valid verbatim.")
    return df


def evaluate(y_true, y_pred, y_true_ord, y_pred_ord):
    return {
        "Macro-F1": f1_score(y_true, y_pred, average="macro"),
        "Balanced Acc": balanced_accuracy_score(y_true, y_pred),
        "QW Kappa": cohen_kappa_score(y_true_ord, y_pred_ord, weights="quadratic"),
    }


def main():
    df = load_data()

    # Re-split this verbatim-augmented subset into its own train/test,
    # since we need a train fold to fit the text classifier too.
    train_df, test_df = train_test_split(
        df, test_size=0.3, stratify=df["NPS_Category"], random_state=42
    )

    y_train, y_test = train_df["NPS_Category"], test_df["NPS_Category"]
    y_train_ord = y_train.map(TARGET_MAP_NUM)
    y_test_ord = y_test.map(TARGET_MAP_NUM)

    # --- 1. Tabular-only baseline ---
    X_train_tab = train_df.drop(columns=["Customer ID", "NPS_Category", "verbatim"])
    X_test_tab = test_df.drop(columns=["Customer ID", "NPS_Category", "verbatim"])
    cat_cols, num_cols = split_column_types(X_train_tab)
    prep_tab = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])
    pipe_tab = Pipeline([("prep", prep_tab), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    pipe_tab.fit(X_train_tab, y_train)
    pred_tab = pipe_tab.predict(X_test_tab)
    pred_tab_ord = pd.Series(pred_tab).map(TARGET_MAP_NUM)

    # --- 2. Text-only ---
    print("Encoding verbatims with sentence-transformers (all-MiniLM-L6-v2)...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    emb_train = encoder.encode(train_df["verbatim"].tolist(), show_progress_bar=False)
    emb_test = encoder.encode(test_df["verbatim"].tolist(), show_progress_bar=False)

    clf_text = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf_text.fit(emb_train, y_train)
    pred_text = clf_text.predict(emb_test)
    pred_text_ord = pd.Series(pred_text).map(TARGET_MAP_NUM)

    # --- 3. Tabular + text combined ---
    X_train_tab_enc = pipe_tab.named_steps["prep"].transform(X_train_tab)
    X_test_tab_enc = pipe_tab.named_steps["prep"].transform(X_test_tab)
    if hasattr(X_train_tab_enc, "toarray"):
        X_train_tab_enc = X_train_tab_enc.toarray()
        X_test_tab_enc = X_test_tab_enc.toarray()

    X_train_combined = np.hstack([X_train_tab_enc, emb_train])
    X_test_combined = np.hstack([X_test_tab_enc, emb_test])

    clf_combined = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf_combined.fit(X_train_combined, y_train)
    pred_combined = clf_combined.predict(X_test_combined)
    pred_combined_ord = pd.Series(pred_combined).map(TARGET_MAP_NUM)

    # --- Results ---
    results = {
        "Tabular only": evaluate(y_test, pred_tab, y_test_ord, pred_tab_ord),
        "Text only": evaluate(y_test, pred_text, y_test_ord, pred_text_ord),
        "Tabular + Text": evaluate(y_test, pred_combined, y_test_ord, pred_combined_ord),
    }
    res_df = pd.DataFrame(results).T
    print("\n=== Results ===")
    print(res_df.round(3))
    res_df.to_csv("data/processed/text_signal_evaluation.csv")

    delta_f1 = results["Tabular + Text"]["Macro-F1"] - results["Tabular only"]["Macro-F1"]
    print(f"\nMacro-F1 delta from adding text: {delta_f1:+.3f}")
    if abs(delta_f1) < 0.02:
        print("-> The text signal adds negligible value over the tabular baseline on this sample.")
    elif delta_f1 > 0:
        print("-> The text signal provides a measurable improvement.")
    else:
        print("-> Adding text actually hurts performance here (possibly noise/small sample).")


if __name__ == "__main__":
    main()