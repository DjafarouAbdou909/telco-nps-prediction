# Telco NPS Prediction

![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8+-F7931E?logo=scikitlearn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-explainability-8A2BE2)
![Gradio](https://img.shields.io/badge/Gradio-interface-FF7C00?logo=gradio&logoColor=white)


A machine learning system that predicts Net Promoter Score (NPS) category — Detractor, Passive, or Promoter — for telecom customers who have not responded to an NPS survey, using account, service, and billing data. The system is designed to support a retention team's outreach prioritization, not to replace human judgment about individual customers.

## Business Context

Net Promoter Score is the standard loyalty metric used by telecom operators to track customer sentiment, but survey response rates are structurally low: in the scenario this project targets, only about 15% of the customer base ever answers an NPS survey. This creates a coverage gap that matters operationally — a retention team cannot act on a metric it does not have for 85% of its customers.

The role of this system is to close that gap by predicting NPS category for the silent majority from data the business already has: tenure, contract terms, subscribed services, billing amount, and payment behavior. The output feeds a retention workflow with two concrete uses: ranking customers by predicted Detractor risk to prioritize outreach, and surfacing the features driving a given prediction so a retention agent has something actionable to discuss, not just a label.

Two properties of this problem shape every downstream decision in the repository. First, NPS is fundamentally ordinal — Detractor, Passive, and Promoter are not interchangeable categories, and a model that confuses Detractor with Promoter is making a qualitatively worse error than one that confuses Detractor with Passive. Second, the label is noisy by construction: this project derives NPS from a five-point Satisfaction Score rather than observing true NPS directly, which means the target itself carries mapping decisions that must be documented and stress-tested rather than treated as ground truth.

## Repository Structure

```
telco-nps-prediction/
├── src/
│   ├── __init__.py
│   ├── features.py
│   ├── model.py
│   └── explain.py
├── app.py
├── train_and_save_models.py
├── test_model_loading.py
├── generate_verbatims.py
├── evaluate_text_signal.py
├── data/
│   ├── raw/
│   │   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   │   └── Telco_customer_churn.xlsx
│   └── processed/
│       ├── telco_nps_prepared.csv
│       ├── telco_nps_train.csv
│       ├── telco_nps_test.csv
│       ├── verbatims.csv
│       └── verbatim_prompt.txt
├── models/
│   ├── nps_model_logreg.joblib
│   ├── nps_model_gradientboosting.joblib
│   ├── nps_model_ordinal.joblib
│   └── model_metadata.json
├── notebooks/
│   ├── 02_data_understanding.ipynb
│   ├── 03_data_preparation.ipynb
│   ├── 04_modeling.ipynb
│   ├── 05_evaluation.ipynb
│   ├── 06_interpretability.ipynb
│   ├── 07_fairness_audit.ipynb
│   └── 09_productization.ipynb
└── requirements.txt
```

`src/` holds the code that must behave identically wherever it runs — the preprocessing pipeline, the model definition, and the SHAP explainer setup. Everything in `src/` is imported, never copy-pasted, by both the notebooks and the production entry points (`app.py`, `train_and_save_models.py`). This separation exists because the project initially duplicated pipeline construction code across five notebooks for the sake of notebook self-containment; the duplication was a correct choice for exploratory notebooks (each one should run standalone, top to bottom, without depending on another notebook's kernel state) but a liability for anything that needs to match production behavior exactly. `src/` is the single source of truth for that behavior.

`notebooks/` is organized by pipeline stage, one notebook per CRISP-DM phase, each of which is independently reproducible: it reloads data from `data/processed/`, reconstructs whatever model or artifact it needs, and asserts its own preconditions rather than trusting upstream state. Notebook 08 (geographic feature engineering, using ZIP/latitude/longitude from the IBM Telco Location table) is deliberately absent — the Location file was not available in this project, and rather than skip the number silently, its omission is documented here and in `03_data_preparation` as a known scope gap, not an oversight.

`data/raw/` contains the two IBM Telco Customer Churn source files as delivered. `data/processed/` contains everything derived from them: the cleaned, feature-engineered, leakage-free modeling table, its train/test split, and the artifacts from the optional text-signal track (verbatims and the prompt template used to generate them).

`models/` contains persisted, ready-to-load model artifacts plus a metadata file recording exactly which library versions produced them, which columns were used, and which columns were deliberately excluded. This metadata file exists because of a concrete failure encountered during development: a model pickled under scikit-learn 1.8 failed to unpickle under 1.9 with an internal `No module named '_loss'` error, since `HistGradientBoostingClassifier` references private submodules that are not stable across versions. The fix was not to patch the pickle but to stop treating pickled binaries as portable across environments — `train_and_save_models.py` is meant to be re-run locally, not have its output copied between machines with different dependency versions.

## End-to-End Pipeline

**Data ingestion.** Two IBM Telco Customer Churn files are merged on customer ID: a 21-column demographics/services/billing table and an 11-column status table containing Satisfaction Score, a pre-computed Churn Score, CLTV, and churn category/reason. The merge is validated as one-to-one — both files cover the same 7,043 customers with no orphans on either side, and `Churn` in the first file is checked to agree with `Churn Label` in the second on every row before proceeding.

**Target construction.** NPS category is derived from Satisfaction Score, since no ground-truth NPS exists in the source data. This is treated as a first-class engineering decision, not a preprocessing footnote — see Target Engineering below.

**Preprocessing.** `TotalCharges` is parsed to numeric (it arrives as a string with blanks for zero-tenure customers) and its 11 blank values are set to 0.0 rather than imputed with a summary statistic, since a customer with zero tenure has genuinely billed nothing yet. Two constant columns (`Count`, `Quarter`) are dropped as uninformative. Outlier screening via IQR on all numeric columns found no true outliers on the continuous variables (`tenure`, `MonthlyCharges`, `TotalCharges`, `CLTV`); it did flag Satisfaction Score itself, which is a known false positive of applying an IQR test to a bounded five-point ordinal variable rather than a continuous one, and is documented as such rather than acted on.

**Feature engineering.** Four features are derived beyond the raw columns: a count of subscribed services, a monthly-charge-per-service ratio, a household-size proxy from partner/dependent flags, and a binary automatic-payment flag. Each is justified individually in Feature Engineering below.

**Train/validation split.** An 80/20 stratified split on NPS category. The stratification keeps class proportions identical between folds, but it does not — and cannot, given this dataset — simulate the actual production gap between the 15% of customers who answer surveys and the 85% who do not, since Satisfaction Score is populated for 100% of customers here. This is stated explicitly as a limitation rather than implied to be solved; a production deployment would need to monitor for distribution shift between historical survey respondents and the full customer base.

**Model training.** Three model families are trained and compared rather than one: a class-weighted logistic regression baseline, a histogram gradient boosting classifier, and an ordinal logistic model. See Modelling Strategy.

**Hyperparameter optimization.** Deliberately out of scope for this iteration. Each model uses reasonable defaults plus class-weighting to handle imbalance; the project's evaluation effort went into comparing model families and metrics rather than tuning a single family, which is a scope trade-off consistent with a two-week time-box and stated explicitly in Future Improvements.

**Evaluation.** Multiple metrics suited to an ordinal, imbalanced, three-class target — not accuracy — plus calibration curves and lift curves specific to the Detractor class, since that is the class the business acts on.

**Explainability.** SHAP values computed directly on the deployed production model (not a separate proxy model), covering global drivers, segment-level effects, and per-customer local explanations.

**Fairness audit.** Detractor-class recall compared across demographic subgroups, with root-cause investigation of the disparity found, not just its detection.

**Inference.** A trained pipeline exposed through both a persisted `.joblib` artifact and a Gradio interface for interactive use.

**Deployment artifacts.** `train_and_save_models.py` produces the three model files and a metadata JSON; `app.py` loads (or, in its current form, retrains) the production model and serves predictions with explanations.

## Dataset

The source is the IBM Telco Customer Churn dataset (version 11.1.3+), split across two files in this project: a demographics/services/billing table and a status table. Both cover the same 7,043 fictional California telecom customers.

Notable variables include `tenure` (months as a customer), `Contract` (month-to-month, one year, two year), `InternetService`, a set of binary service flags (`OnlineSecurity`, `TechSupport`, `StreamingTV`, and similar), `MonthlyCharges`/`TotalCharges`, `PaymentMethod`, and — critically for this project — `Satisfaction Score`, a five-point scale that is the only available proxy for NPS.

The central assumption of this project is that Satisfaction Score, though not literally NPS, is close enough in construct to serve as its basis, per the challenge brief that motivated this work. This is an assumption, not a fact verified against real NPS data, and is treated as such throughout.

Two variables carry a documented leakage risk that shaped the entire feature-selection process: `Churn Score`, a pre-computed risk score whose mean differs by nearly 32 points between customers who ultimately churned and those who did not, and `Churn Value`, the realized churn outcome itself. Both are excluded from every model's feature set. A third variable, `CLTV`, was checked for the same risk and found clean (no meaningful relationship to Satisfaction Score) but is excluded anyway out of caution, since its proprietary calculation is undocumented by IBM and may itself embed churn-risk information not visible in this dataset.

A geographic dimension (ZIP code, latitude/longitude) exists in the broader IBM Telco dataset but was not obtained for this project. Its absence is a real limitation for the fairness audit in particular, since ZIP code is a commonly cited proxy for socioeconomic status; this project can state that no such proxy was used, but cannot claim to have audited for one it never had access to.

## Target Engineering

NPS category is reconstructed from Satisfaction Score because no direct NPS field exists in the source data. Two mappings were built and compared rather than committing to the first one tried.

The baseline mapping — Satisfaction 5 to Promoter, 4 to Passive, 1 through 3 to Detractor — produces a 58.3% Detractor rate, which is high enough to be suspicious for a real NPS distribution and is a direct artifact of collapsing three of five satisfaction levels into a single class.

The refined mapping treats a Satisfaction Score of 3 as genuinely ambiguous and resolves it using realized churn behavior: a score-3 customer who subsequently churned is labeled Detractor, one who stayed is labeled Passive. This produces a 26.5% Detractor rate that matches the dataset's true churn rate almost exactly, which is the primary justification for preferring it — a coincidence this precise between an independently-derived label and ground-truth churn behavior is a stronger validation signal than an arbitrary threshold choice would provide.

The two mappings disagree on 31.7% of customers, which means the choice is consequential, not cosmetic, and both are retained in the codebase for sensitivity comparison rather than only the final choice.

Using `Churn Value` to resolve the ambiguous case is a target-construction decision, not a feature-engineering one, and the distinction is enforced mechanically: `Churn Value`, along with `Churn Score`, `Churn Label`, `Satisfaction Score` itself, and the intermediate `NPS_baseline` comparison column, are all excluded from every model's input features. This exclusion was not correct on the first attempt — an earlier version of the pipeline retained `NPS_baseline` in the modeling table by omission, where it dominated every downstream SHAP analysis with a global importance roughly double that of the next feature, since it agrees with the true target on 68.3% of rows. The bug was caught by comparing notebook output against a fresh execution rather than trusting a single run, and the fix is now enforced by an explicit assertion at the top of every downstream notebook (`assert "NPS_baseline" not in train.columns`) rather than relying on the drop list alone.

## Feature Engineering

Twenty-four features feed the production model, organized into five families.

Demographic features (`gender`, `SeniorCitizen`, `Partner`, `Dependents`) are used directly rather than through a derived proxy, which is what makes the fairness audit possible — an audit needs the protected attribute to be visible, not laundered through an intermediate feature.

Account and contract features (`tenure`, `Contract`, `PaymentMethod`, `PaperlessBilling`) carry the strongest global signal in this dataset. `Contract` in particular is both highly predictive and directly actionable by the business, which matters for how the model's output gets used downstream — a driver the retention team can act on is worth more operationally than one they cannot.

Service usage features are the eight binary service flags plus `InternetService`, combined into an engineered `n_services` count (0 to 8) capturing overall service engagement independent of any single service.

Financial features are `MonthlyCharges` and `TotalCharges`, plus one engineered ratio, `charges_per_service`, built specifically to separate two customer profiles that raw `MonthlyCharges` conflates: a customer paying a lot for few services and a customer paying a similar amount for many services. These have different risk profiles that the ratio helps distinguish.

Engagement proxies are `is_autopay` (derived from `PaymentMethod`, since automatic-payment customers show consistently lower dissatisfaction across this dataset) and `household_size_proxy` (the sum of `Partner` and `Dependents`), included per the challenge brief's suggestion of household-composition signal.

Geographic and interaction features were scoped out for the reason stated above — no Location data was available — and no synthetic geographic feature was fabricated to fill the gap.

## Modelling Strategy

Three families are trained and compared rather than one, because the brief's central modeling question — how to frame an ordinal target — does not have a single correct answer, and comparing families is how that question gets an evidence-based answer instead of an assumed one.

The baseline is a class-weighted logistic regression, included specifically because skipping a disciplined baseline tends to produce overconfidence in whatever comes next. `class_weight="balanced"` is applied from this first model onward to counteract the roughly 3.5:1 imbalance between the largest and smallest NPS classes.

The second family is a histogram gradient boosting classifier (`HistGradientBoostingClassifier` from scikit-learn), chosen over an external library such as LightGBM or XGBoost because it handles categorical features natively without manual encoding and introduces no additional dependency, while still representing the gradient boosting family the brief asks for.

The third family is an ordinal logistic model (`mord.LogisticAT`), included specifically to test whether respecting the ordinal structure of the target changes the outcome rather than assuming it does. This model surfaced a real tooling gap during development: `mord` does not support `class_weight` natively, and without a manual `sample_weight` correction it collapses entirely onto the majority class, predicting zero Promoters. The corrected version, using `compute_sample_weight`, is the one reported throughout.

None of the three models dominates on every metric, and the final model selection changed once between the modeling notebook and the evaluation notebook as a direct result of measuring calibration and lift rather than stopping at classification metrics. The ordinal model has the best quadratic weighted kappa and by far the fewest severe misclassifications (Detractor predicted as Promoter, or the reverse) — 2.6% of predictions versus 8.5% to 10.0% for the other two. But on lift — the fraction of true Detractors captured within the top 20% of customers ranked by predicted risk, which is the metric that maps most directly onto "how many detractors does the retention team reach by calling the top of the list" — logistic regression leads clearly, capturing 47.6% of true Detractors in the top 20% of ranked customers versus 42.0% for the ordinal model. Since the stated business objective is outreach prioritization, logistic regression is the model selected for production, and this reversal from the modeling notebook's initial preference is recorded explicitly rather than silently overwritten.

None of the three models is well-calibrated in an absolute sense — class weighting, which is necessary to get any signal on the minority classes at all, distorts predicted probabilities away from true frequencies as a direct side effect. This is why the production interface displays a calibration caveat next to every prediction: the probabilities are reliable for ranking customers against each other, not for reading as literal risk percentages.

## Evaluation

Accuracy is not reported as a primary metric because it is actively misleading on this target: a model that always predicts the majority class (Passive) would score 57% accuracy while learning nothing.

**Macro-F1** averages F1 across all three classes unweighted by their size, which is why it is reported instead of a size-weighted average — it penalizes a model for ignoring the minority classes even if the majority class is handled well, which is exactly the failure mode this project needs to avoid given that Detractor is both a minority class and the one the business cares most about.

**Balanced accuracy** (the average of per-class recall) is reported alongside Macro-F1 as a second, differently-shaped view of the same imbalance problem, since the two metrics can disagree when precision and recall trade off differently across classes.

**Quadratic weighted kappa** is the metric that actually encodes the ordinal structure of NPS: it penalizes a Detractor-predicted-as-Promoter error more heavily than a Detractor-predicted-as-Passive error, which none of the other metrics used here do. It is the deciding factor in why the ordinal model was seriously considered for production despite a lower Macro-F1.

**Per-class recall on Detractor** is reported separately because it is the number that most directly answers the business question of the project: of the customers the retention team should call, how many does the model actually surface. It is also the metric behind the fairness audit, applied per demographic subgroup rather than in aggregate.

**Confusion matrices** are read specifically for the count of far errors — predictions two classes away from the truth — since this count is what separates a model making forgivable mistakes from one making costly ones, and this distinction is invisible in any single scalar metric.

**Calibration curves** compare predicted probability against observed frequency within probability bins, and found meaningful miscalibration in all three models, driven by class weighting.

**Lift curves** rank the test set by predicted Detractor probability and measure what fraction of true Detractors are captured within the top X% — the single most business-relevant curve in this evaluation, since it mirrors exactly how a retention team would use the model's output: call down a ranked list.

## Explainability

SHAP is computed with `shap.LinearExplainer` directly on the deployed logistic regression pipeline, not a separate tree-based proxy model, so that the explanation is guaranteed to match the model actually making predictions.

This choice surfaced a reproducibility defect worth documenting: `LinearExplainer`'s default masker subsamples its background data to 100 rows without a fixed seed, which means SHAP values were not reproducible across environments even with identical library versions — two runs of the same notebook in two different setups produced measurably different top-driver rankings. The fix is an explicit `shap.maskers.Independent` instantiated with `max_samples` equal to the full training set size, removing the subsampling entirely; this is small enough a dataset that the computational cost of doing so is negligible.

Global feature importance is dominated by `tenure`, followed by billing amount and contract type, consistent across both the linear model's SHAP values and the gradient boosting model's permutation importance computed independently — two different explanation methods agreeing is treated as a robustness check, not a redundant computation.

Segment-level analysis shows that the sign of a feature's effect can fully invert depending on customer segment: `tenure` pushes toward Detractor for new customers and away from it for long-tenured ones, and the same reversal holds for contract type and internet service type. This is presented as a reason not to act on a single global rule ("target fiber customers") without conditioning on the rest of a customer's profile.

One finding is highlighted specifically as a correlation-versus-causation caveat: `OnlineSecurity` shows a lower raw Detractor rate among subscribers (14.3% versus 41.2% for non-subscribers) but a positive SHAP contribution toward Detractor once the model's other coefficients — chiefly `tenure` and `Contract`, which are both strongly associated with `OnlineSecurity` adoption — are accounted for. The explanation offered is confounding, not a claim that subscribing to online security causes dissatisfaction, and the repository is explicit that no retention action should be built on this signal alone without a controlled test.

An actionable-versus-non-actionable classification is maintained for every feature the model uses, since the strongest global driver (`tenure`) is also entirely non-actionable — it cannot be changed retroactively — which means the model's best predictor and the business's best lever are not the same thing, and conflating them would misdirect a retention team's effort.

## Fairness

Detractor-class recall is compared across `gender`, `SeniorCitizen`, `Partner`, and `Dependents`, since these are the demographic attributes present and used directly as features in this dataset.

`gender` shows no meaningful disparity (70.4% recall for female customers, 71.9% for male). `SeniorCitizen` shows a real disparity, but in the opposite direction from what was hypothesized during data exploration — senior customers are captured better (88.6% recall) than non-senior customers (66.4%), not worse, and this reversal from the initial hypothesis is reported as found rather than adjusted to match expectation.

`Dependents` is the material finding: recall drops to 51.4% for customers with dependents against 75.8% for those without, a 24-point gap comparable in magnitude to the disparity example given in the original brief. Root-cause investigation attributes this to confounding with the model's two strongest features — customers with dependents have both longer average tenure and a much lower rate of month-to-month contracts, both of which the model reads as low-risk signals, independent of whether the customer is actually a Detractor. This produces a concrete downstream failure mode: 29% of true Detractors with dependents are misclassified as Promoters (the worst possible error), versus 15.5% for customers without dependents.

No ZIP-code or other geographic proxy audit was performed, because no geographic data was available to this project, and this is stated as a scope gap in the audit rather than implied to be covered.

The recommendation attached to this finding is deliberately not "drop the feature" — `Dependents` is not noise, and removing it would cost real predictive performance elsewhere. The recommendation is escalation to a Customer Experience or Legal review before this model is used to allocate retention budget, alongside a differentiated decision threshold for the affected subgroup rather than a blanket model change, and an explicit warning against naively reweighting the training data without first addressing the underlying confound, which risks degrading performance for the majority group without fixing the actual problem.

## Model Persistence

Models are serialized with `joblib`, the standard for scikit-learn-based pipelines, alongside a `model_metadata.json` recording the scikit-learn and `mord` versions used, the exact feature column list, the columns explicitly excluded for leakage, and the reason the primary model was selected.

Versioning here is handled by not treating pickled artifacts as portable: `train_and_save_models.py` is the source of truth, meant to be executed in the target environment rather than have its output copied across environments with different dependency versions, following the cross-version unpickling failure described earlier. Loading is verified by `test_model_loading.py`, which reloads each artifact in a fresh process and checks that its exposed interface (`predict_proba` or the equivalent for the ordinal model's stored preprocessor-plus-model dict) is intact before it is trusted.

## Inference

A prediction request is a single customer's row of raw feature values — demographics, contract, services, billing — either looked up by customer ID from the test set or entered manually. The pipeline handles preprocessing (one-hot encoding with `handle_unknown="ignore"`, so a category never seen during training does not raise an exception but is instead encoded as absent across all known categories) and returns three things together: the predicted class, the full probability distribution across all three classes, and the top five SHAP-ranked features driving that specific prediction toward or away from Detractor. Confidence scores are the raw class probabilities, presented with the calibration caveat noted above rather than as literal risk percentages.

## Gradio Application

The interface is built with Gradio rather than Streamlit, chosen because it launches directly from a notebook context during development (`prevent_thread_lock=True`) and converts to a standalone script without code changes, which matched this project's iterate-then-deploy workflow.

The application exposes two tabs. The first looks up an existing test-set customer by ID and returns a prediction with its local SHAP explanation. The second accepts raw manual input across all demographic, contract, service, and billing fields and reconstructs the same engineered features (`n_services`, `charges_per_service`, `household_size_proxy`, `is_autopay`) that the training pipeline computes, so a hypothetical customer is scored identically to a real one rather than through a simplified parallel path.

Robustness was verified directly rather than assumed: an unseen `PaymentMethod` value was passed through the manual-entry path and confirmed not to raise an exception, and the existing-customer lookup was checked against the corresponding `06_interpretability` analysis to confirm the interface reproduces the same prediction and driver ranking as the notebook, not a silently divergent version.

The stated limitation is scope, not a defect: the bonus feature of pasting a free-text customer verbatim and observing its effect on the prediction is not implemented in the current interface, since the verbatim-generation track (see below) is a separate, optional extension not yet integrated into `app.py`.

## Installation

Requires Python 3.12.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Place the two raw IBM Telco files in `data/raw/`:

```
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
data/raw/Telco_customer_churn.xlsx
```

Run the pipeline notebooks in order (02 through 07, then 09) to produce `data/processed/telco_nps_train.csv` and `telco_nps_test.csv`. Train and persist the models:

```bash
python train_and_save_models.py
```

Launch the interface:

```bash
python app.py
```

The optional text-signal track requires a Gemini API key:

```bash
export GEMINI_API_KEY="your-key-here"
python generate_verbatims.py
python evaluate_text_signal.py
```

## Configuration

There is no separate configuration file in the current version of this repository; configuration lives as named constants at the top of each script (`SEED`, `NOISE_FLIP_RATE`, and `MODEL_NAME` in `generate_verbatims.py`; the drop-column list in `src/features.py`) rather than in an external YAML or JSON file, which is a deliberate simplification for a project at this scale rather than an oversight. Externalizing these into a single configuration file is listed under Future Improvements.

## Reproducibility

Random seeds are fixed wherever randomness affects a result that gets reported: `random_state=42` for the train/test split and both gradient boosting and ordinal models, and a separate documented seed for the noise injected into synthetic verbatim sentiment. Preprocessing is deterministic by construction — `OneHotEncoder` and `StandardScaler` have no stochastic component — with the one documented exception being SHAP's background sampling, which was found non-deterministic under default settings and fixed by disabling subsampling entirely rather than merely seeding it, since a seed alone would not guarantee identical results across different `shap` library versions.

Dependency versions are not yet pinned in `requirements.txt` beyond what is noted inline in this document; this is flagged as a gap rather than presented as solved, given that a real cross-version failure (the `HistGradientBoostingClassifier` unpickling error) was directly caused by an unpinned scikit-learn version during development. No experiment tracking system (MLflow, Weights and Biases) is used; results in this repository are reproduced by re-running the relevant notebook, which is sufficient at this project's current scale but would not scale past a handful of model variants.

## Future Improvements

Target construction would benefit from access to real NPS survey data, even a small labeled sample, to validate the Satisfaction Score-based mapping against ground truth rather than against internal consistency alone.

Geographic features (ZIP code, latitude/longitude) were scoped out for lack of source data; obtaining the IBM Telco Location table would both add predictive signal and close the fairness-audit gap around geographic proxies.

Dependency pinning and a lockfile would remove the class of cross-version failures already encountered once in this project. A CI pipeline running `test_model_loading.py` and a smoke test of `app.py` on every change would catch the same class of failure automatically rather than requiring a user to hit it manually.

Containerization (Docker) would remove environment drift as a failure mode entirely, which is a stronger fix than dependency pinning alone for the unpickling issue described above.

The fairness disparity found on `Dependents` warrants a dedicated mitigation pass — most plausibly a segment-specific decision threshold — rather than the blanket recommendation this repository currently documents.

A monitoring layer tracking prediction drift and input distribution drift between historical training data and newly scored customers would directly address the unverified assumption that survey respondents and non-respondents share a feature distribution, which this project states as a limitation but does not currently measure.

An automated retraining trigger, tied either to a schedule or to a minimum count of newly labeled customers, would be a natural extension once real production labels start arriving.

## Technical Stack

**scikit-learn** for preprocessing, the logistic regression and gradient boosting models, and all evaluation metrics — chosen as the default, well-supported choice for a tabular problem of this size.

**mord** for the ordinal logistic model, the only actively maintained Python library offering ordinal classification with a scikit-learn-compatible interface.

**SHAP** for model explanation, chosen over LIME for its stronger theoretical grounding (Shapley values) and its exact, non-sampling-based explainer available for linear models specifically.

**Gradio** for the interactive interface, chosen over Streamlit for faster iteration from within a notebook during development.

**sentence-transformers** for embedding synthetic customer verbatims in the optional text-signal track, chosen for running entirely locally without requiring an embedding API call per customer.

**Gemini API** for synthetic verbatim generation, the LLM provider used for this specific project; the generation script is not tied to Gemini architecturally and could be pointed at another provider's API with a small, isolated change.

**joblib** for model serialization, the standard choice for persisting scikit-learn pipelines.

## References

IBM Telco Customer Churn dataset (11.1.3+): https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Pedregosa et al., "Scikit-learn: Machine Learning in Python," Journal of Machine Learning Research, 2011.

Lundberg and Lee, "A Unified Approach to Interpreting Model Predictions" (SHAP), NeurIPS, 2017.

Fairlearn documentation: https://fairlearn.org

Gradio documentation: https://www.gradio.app/docs

TM Forum, "Improving Net Promoter Score using machine learning," 2021.

Kannan et al., "Prediction of Customer Transactional Net Promoter Score (tNPS) Using Machine Learning: A Telecommunication Company Case Study," 2022.