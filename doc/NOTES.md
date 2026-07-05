# NPS Prediction Project – Notes

## 1. Project Overview

This project aims to predict the NPS category (Detractor, Passive, Promoter) of telecom customers using structured data related to their account and behavior.

The goal is to support customer retention strategies by identifying at-risk customers and understanding the drivers of dissatisfaction.

---

## 2. Business Problem

Only 15% of customers respond to NPS surveys, which creates a significant blind spot for the company.

The challenge is to extend this visibility to the entire customer base using machine learning.

---

## 3. Machine Learning Problem

This is a supervised classification problem with three ordered classes:

- Detractor
- Passive
- Promoter

The target variable is built from the Satisfaction Score (1–5).

---

## 4. Key Assumptions

- The Satisfaction Score is an acceptable proxy for NPS.
- Survey respondents are partially representative of the full customer base.
- Structured features contain enough information to predict customer satisfaction.

---

## 5. Risks and Limitations

- Risk of data leakage with Churn Score and Churn Value.
- Class imbalance (especially Detractors).
- Uncertainty in how the target variable is constructed.
- Distribution shift between respondents and non-respondents.
- Correlation ≠ causation.

---

## 6. Success Criteria

- Good recall on the Detractor class.
- Use of balanced metrics (F1 macro, weighted kappa, etc.).
- Interpretable model.
- Results usable by non-technical teams.

---

## 7. Next Steps

- Data cleaning and preparation
- Feature engineering
- Baseline model
- Advanced modeling
- Evaluation and fairness analysis
- Deployment prototype