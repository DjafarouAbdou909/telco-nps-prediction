# Note: 03 Data Preparation

## Notebook Objective

Build the NPS target from `Satisfaction Score`, clean data quality issues identified in `02_data_understanding`, engineer initial features, and prepare a train/test split aligned with the business problem prior to any modeling.

## NPS Target Construction

Two mappings were compared:

| | Baseline mapping (brief) | Final mapping |
|---|---|---|
| Rule | Score ≤3 → Detractor, 4 → Passive, 5 → Promoter | Score 3 handled case-by-case using `Churn Value` (churned → Detractor, retained → Passive) |
| Detractors | 58.3% | **26.5%** |
| Passives | 25.4% | **57.1%** |
| Promoters | 16.3% | 16.3% |


**Decision: refined mapping retained.** The main justification is that the resulting detractor rate (26.5%) exactly matches the real churn rate in the dataset (`Churn Value` = 1 for 26.5% of customers), which provides strong business consistency. In contrast, the baseline mapping produces an artificially detractor-heavy distribution (58%), which is atypical for a realistic NPS setting.

31.7% of customers change category between the two mappings (sensitivity analysis), meaning this choice is not neutral and should be revalidated during modeling (including training on the baseline mapping to ensure conclusions are not overly dependent on thresholding).

**Target construction vs feature leakage distinction is explicitly documented:** `Churn Value` is used to *define* the target (valid product decision a priori), but is **never** used as an input feature. `Satisfaction Score`, `Churn Score`, `Churn Value`, and `Churn Label` are all excluded from the feature set including them would make the problem trivial on observed customers but unusable on the 85% silent customers targeted in production.

## Cleaning performed

- `TotalCharges`: converted from `str` → `float`, 11 missing values (all `tenure = 0`) replaced with `0.0`
- Constant columns removed: `Count`, `Quarter`
- Leakage / post-hoc columns excluded from modeling set: `Satisfaction Score`, `Customer Status`, `Churn`, `Churn Label`, `Churn Value`, `Churn Score`, `CLTV`, `Churn Category`, `Churn Reason`

`CLTV` was validated as safe during Data Understanding, but excluded as an additional precaution (unobserved risk component in its construction). It may be reintroduced and tested later during modeling if performance trade-offs justify it.

## Feature Engineering

| Feature | Description |
|---|---|
| `n_services` | Number of subscribed services (0–8) |
| `charges_per_service` | `MonthlyCharges` / `n_services` — isolates perceived price per service |
| `household_size_proxy` | `Partner` + `Dependents` (0–2) |
| `is_autopay` | Automatic payment flag |

A total of 24 features were retained for the modeling set (`Customer ID` kept for traceability/interface purposes, but excluded from numerical model inputs).

Geographical features (ZIP/Lat/Long) were **not engineered** due to the absence of the IBM Location file. This is an acknowledged scope limitation, not an oversight.

## Validation Strategy

A stratified 80/20 split on `NPS_Category` was applied (train: 5634 rows, test: 1409 rows), where the test set simulates the "silent customers" that the model never observes during training.

**Documented limitation:** this split assumes that actual NPS survey respondents (15% in reality) share the same feature distribution as non-respondents (85%). This is an optimistic assumption and may not hold in production (selection bias: customers who respond are often more engaged).

## Class Imbalance

Majority/minority ratio (Passive/Promoter): **3.5**. Implication for `04_modeling`: accuracy alone is misleading (a "always Passive" model already achieves 57%) — therefore macro-F1, balanced accuracy, or quadratic weighted kappa (which accounts for class ordering) should be used, with a particular focus on recall for the Detractor class.

## What is deferred to `04_modeling`

- Problem framing choice (classification / ordinal classification / regression + thresholding), to be explicitly justified
- A disciplined baseline model (logistic regression or gradient boosting as default)
- Comparison of at least two model families
- Retraining using the baseline mapping to test target sensitivity
- Selection and computation of appropriate metrics (macro-F1, balanced accuracy, quadratic weighted kappa, per-class recall)