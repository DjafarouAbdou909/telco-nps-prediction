# Note — 06 Interpretability

**Project**: Artefact Take-Home Challenge — Customer NPS Prediction  
**Associated Notebook**: `06_interpretability.ipynb`  
**Status**: Completed (corrected twice — see Limitations)

## Notebook Objective

Apply SHAP analysis to the actual production model (Logistic Regression, not a proxy model), including global drivers, segment-level effects, actionable vs non-actionable factors, and a recommendation function — section 4.6 of the challenge description.

## Reproducibility Bug Fixed

`shap.LinearExplainer` silently samples its background dataset to 100 rows by default, without a guaranteed seed across environments. As a result, two apparently identical executions could produce different feature rankings.

**Fixed** by using an explicit masker (`shap.maskers.Independent`) with `max_samples` set to the actual training set size.

All notebook values were recalculated after this fix. This explains why the recommendation for the example customer changed (`InternetService` instead of the initially reported `OnlineSecurity`).

## Global Drivers (Detractor Class)

`tenure` (0.804) is by far the strongest driver, followed by:
- `TotalCharges`
- `MonthlyCharges`
- `Contract_Month-to-month`
- `InternetService_Fiber optic`

These results are consistent with the permutation importance from the Gradient Boosting model in `04_modeling` — two independent methods leading to the same conclusion.

## Segment-Dependent Effects

The direction of the effect completely changes depending on the customer profile:

- `tenure`: new customers (+1.11) vs long-term customers (-1.13)
- `Contract`: monthly contract (+0.20) vs yearly contract (-0.23)
- `InternetService`: fiber (+0.24) vs DSL (-0.19)

**Implication**: there is no valid universal business rule such as "target all fiber customers" without considering the complete customer profile.

## Correlation vs Causation Discovery — OnlineSecurity

Raw Detractor rate:
- `OnlineSecurity=Yes`: 14.3%
- `OnlineSecurity=No`: 41.2%

At first glance, online security appears protective.

However, SHAP shows the opposite direction once `tenure` and `Contract` effects are controlled for: the partial effect of `OnlineSecurity=Yes` is positive toward Detractor.

Explanation: customers with `OnlineSecurity=Yes` are structurally older customers and less likely to have monthly contracts. This reflects feature correlation, not causation.

No business action should be based on this signal alone without a controlled A/B test.

## Actionable vs Non-Actionable Drivers

`tenure` (the strongest driver) is **non-actionable**.

The notebook explicitly distinguishes between:
- the "best predictor" of risk;
- the "best business action lever".

Priority actionable features include:
- `Contract`
- `PaymentMethod`
- Additional services

## Limitations

Two major corrections were applied:

1. The `NPS_baseline` data leakage issue (inherited from `03_data_preparation`) was fixed upstream.
2. The SHAP background sampling issue was fixed in this notebook.

Both corrections changed the reported values. The final version contains internally consistent results, verified through a complete re-execution.

## Deferred to `07_fairness_audit`

Evaluate whether the disparity signal identified for `SeniorCitizen` during `02_data_understanding` translates into a measurable difference in Detractor recall.