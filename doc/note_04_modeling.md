# Note — 04 Modeling

## Notebook Objective

Train and compare several model families to predict `NPS_Category`, using a disciplined baseline and metrics adapted to an imbalanced 3-class ordinal target (3.5 ratio between the majority and minority classes).

## Problem Framing

Three model families were compared instead of selecting only one, in order to empirically evaluate how to handle the ordinal nature of the target rather than assuming a specific approach:

- **Baseline**: Logistic Regression, `class_weight="balanced"`
- **Gradient Boosting**: `HistGradientBoostingClassifier` (native scikit-learn implementation, no external dependency such as LightGBM/XGBoost — handles categorical features natively)
- **Ordinal Model**: `mord.LogisticAT`, with manually computed `sample_weight` (the package does not natively support `class_weight` — without this correction, the model completely collapses toward the majority class, with 0% recall for Promoters)

## Results

| Model | Macro-F1 | Balanced Acc | QW Kappa | Far error rate |
|---|---|---|---|---|
| Logistic Regression | 0.513 | 0.592 | 0.346 | 10.0% |
| Gradient Boosting | 0.514 | 0.533 | 0.290 | 8.5% |
| Ordinal (LogisticAT) | 0.475 | 0.481 | **0.365** | **2.6%** |
| Naive baseline (always Passive) | 0.242 | 0.333 | 0.000 | — |

All three models significantly outperform the naive baseline, confirming that a real predictive signal exists in the data.

**No model dominates across all metrics**: the Ordinal Model achieves the best QW Kappa and by far the lowest extreme error rate (Detractor classified as Promoter), while Logistic Regression achieves the highest raw Detractor recall.

**Initial decision (reviewed in `05_evaluation`)**: the Ordinal Model was selected as the reference model based on its QW Kappa score and lower extreme error rate, considered at this stage as the metric most aligned with the business cost of severe NPS classification errors.

## Feature Importance (Gradient Boosting Overview)

Permutation importance shows that `tenure`, `OnlineSecurity`, and `Contract` are the dominant features, while the remaining features contribute much less.

These findings were independently confirmed by SHAP analysis in `06_interpretability`.

## Produced Files

- `04_modeling.md` — complete notebook
- `fig_model_comparison_metrics.png`
- `fig_far_errors.png`
- `fig_confusion_matrices.png`
- `fig_feature_importance.png`

## Deferred to `05_evaluation`

- Calibration plots and lift curves (required by section 4.5 of the challenge description, initially missing from this notebook — issue fixed after review)
- Re-evaluation of the final model choice after measuring calibration and lift performance