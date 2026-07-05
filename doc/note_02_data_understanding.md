
# Note: 02 Data Understanding  

## Notebook Objective

Explore raw data before any transformation in order to assess data quality, understand its structure, and identify risks (data leakage, bias, missing values) before building the NPS target and performing feature engineering — following the CRISP-DM separation between Data Understanding and Data Preparation.

This notebook intentionally applies **no final transformations** (no target encoding, no cleaning): it only documents what needs to be done, without performing it.

## Data Sources

| File | Rows | Columns | Content |
|---|---|---|---|
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | 7043 | 21 | Customer demographics, subscribed services, contract, billing |
| `Telco_customer_churn.xlsx` | 7043 | 11 | Satisfaction Score, Churn Score, CLTV, churn reasons |

Merged on `Customer ID` (`customerID` renamed for consistency), validated as a **1:1 join** — same population in both files, no orphan records. Final dataset: **7043 rows × 31 columns**.

Cross-check validation: `Churn` (services file) == `Churn Label` (status file) for 100% of rows — strong consistency across sources.

## Key Findings

### Data Quality
- **No duplicates**, no real missing values except two cases:
  - `TotalCharges`: 11 empty strings (not detected by `isna()`), all with `tenure = 0` → new customers without first invoice. Should be set to `0.0`, not removed or imputed with mean.
  - `Churn Category` / `Churn Reason`: 5,174 missing values, structural (only filled for churned customers).
- **Constant columns to remove**: `Count` (always 1), `Quarter` (always "Q3").
- **No real outliers** in continuous variables (`tenure`, `MonthlyCharges`, `TotalCharges`, `CLTV`, `Churn Score`) using IQR method. `Satisfaction Score` produces 922 false positives because it is an ordinal variable bounded from 1 to 5 — a known limitation of IQR, not a data issue.

### Data Leakage Risk
Strong quantitative confirmation of leakage highlighted in the problem statement (section 4.1):

- `Satisfaction Score` = 1 or 2 → **100% churn**
- `Satisfaction Score` = 4 or 5 → **0% churn**
- `Churn Score` follows the same pattern (mean ~82 for low satisfaction vs ~50 for high satisfaction)

→ `Churn Score` and `Churn Value` must be excluded from the feature set.  
`CLTV` does not show this deterministic relationship and can be kept.

### Target Distribution Insight
`Satisfaction Score` is centered around score 3 (37.8% of customers). The mapping into 3 NPS categories (Detractor / Passive / Promoter) will strongly impact class balance. Classifying score 3 as “Detractor” (baseline mapping in the challenge) would result in ~58% detractors, which is not realistic for a standard NPS distribution. This design choice must be carefully justified during Data Preparation.

### Promising Features
Consistent and actionable relationships with `Satisfaction Score`:

- `Contract`: month-to-month (2.94) vs two-year (3.67)
- `InternetService`: fiber optic (2.91) vs no internet (3.85)
- `PaymentMethod`: electronic check (2.87) vs others (~3.44)
- `MonthlyCharges`: clear decreasing relationship with satisfaction

### Preliminary Fairness Signal
Senior customers have lower average satisfaction (2.93 vs 3.31) and are overrepresented in low scores (1–2). This should be explicitly monitored during the final fairness audit (section 4.7). `SeniorCitizen` remains a valid feature, unlike proxies such as ZIP code.

## What is deferred to `03_data_preparation`

- Construction and justification of the mapping `Satisfaction Score → NPS_Category`, including sensitivity analysis
- Handling of `TotalCharges` (conversion + fixing 11 empty values)
- Removal of `Count`, `Quarter`, and leakage-prone columns (`Churn Score`, `Churn Value`, `Churn Label`, `Churn Category`, `Churn Reason`)
- Feature engineering (billing ratios, service bundles, categorical encoding)

