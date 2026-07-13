# Note — 05 Evaluation


## Notebook Objective

Complete `04_modeling` with the two missing evaluation components required by section 4.5 of the challenge description: **calibration plots** and **lift curves for the Detractor class**.

This notebook is self-contained (re-trains the three models from scratch and does not depend on the `04_modeling` notebook kernel).

## Calibration

None of the three models is perfectly calibrated. The class weighting strategy (required to obtain meaningful performance on minority classes) shifts the predicted probabilities away from the true class frequencies. This is an expected and documented side effect.

Brier scores (Detractor class):
- Logistic Regression: 0.146
- Gradient Boosting: 0.154
- Ordinal Model: 0.167

**Practical implication**: predicted probabilities should be used to **rank customers** rather than interpreted as absolute risk percentages. This warning is directly included in the interface (`09_productization`).

## Lift Curves — The Result That Changes the Decision

| Model | Top 20% contacted | Top 30% contacted |
|---|---|---|
| **Logistic Regression** | **47.6%** | **64.7%** |
| Gradient Boosting | 46.0% | 61.8% |
| Ordinal (LogisticAT) | 42.0% | 57.0% |
| Random targeting | 20.0% | 30.0% |

All three models significantly outperform random targeting. However, **Logistic Regression achieves the highest lift**, not the Ordinal Model selected in `04_modeling`.

## Revised Final Decision

| Criterion | Best Model |
|---|---|
| Extreme errors (QW Kappa) | Ordinal Model |
| **Lift / outreach prioritization** | **Logistic Regression** |
| Absolute probability calibration | None of the three |

**Production model selected: Logistic Regression.**

This is an intentional change compared to `04_modeling`. The reason is that the explicit business objective of the challenge is outreach prioritization (contacting the top X% highest-risk customers), not minimizing ordinal classification errors in isolation.

The Ordinal Model remains as a comparison reference for the final write-up.


## Deferred to `06_interpretability`

- Full SHAP analysis on the confirmed production model (Logistic Regression)
- Detractor drivers by customer segment (section 4.6 of the challenge description)