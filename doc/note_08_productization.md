# Note — 08 Productization

**Project**: Artefact Take-Home Challenge — Customer NPS Prediction  
**Associated Notebook**: `09_productization.ipynb`  
**Status**: Completed

## Notebook Objective

Persist the selected model and build a simple interface (section 4.8) allowing a retention agent to either select an existing customer or enter a customer profile manually, then receive an NPS prediction with the main drivers behind the prediction.

## Design Choices

- **Gradio instead of Streamlit**: can be launched directly from a notebook for testing and then deployed as a standalone `.py` application without code changes.

- **Two tabs**: existing customer (dropdown using `Customer ID`) and manual input — covering both use cases required by the challenge description.

- **Calibration warning displayed directly in the UI**, not only in the technical documentation — because the target users are non-data-scientists.

- **Local SHAP explanations recalculated for each prediction**, using the same explainer as `06_interpretability`, with no proxy model.

- **Automatic reconstruction of engineered features** (`n_services`, `charges_per_service`, `household_size_proxy`, `is_autopay`) in the manual input tab — users should not have to compute derived features themselves.

## Real Tests Performed

- Backend tested independently before building the UI (existing customer + manual input).
- Robustness to unknown categorical values explicitly tested (`OneHotEncoder(handle_unknown="ignore")`) — no crash occurs.
- Cross-validation with `06_interpretability`: same customer (`9172-ANCRX`), same probabilities, same drivers — no silent divergence.
- Gradio server actually launched and tested (not only written), both tabs queried, then properly closed.
- Functionality confirmed under real usage conditions by the user (screenshot showing prediction, probabilities, and SHAP chart displayed correctly).

## Refactoring into `src/`

Duplicated code across `04/05/06/07/09` (pipeline, SHAP logic) was extracted into a dedicated `src/` package:

- `features.py`
- `model.py`
- `explain.py`

`app.py` and `train_and_save_models.py` were refactored to import from `src/` instead of redefining the pipeline.

Refactoring validation: predictions remained exactly identical compared to the previous implementation, up to decimal precision.

## Bug Fixed During Development

Incorrect relative path (`data/processed/...` without `../`) in the notebook (located inside `notebooks/`) was fixed to use:

`../data/processed/...`

This is consistent with the other notebooks.

`app.py` keeps the path without `../` because it is launched from the proje

## Explicit Limitation

The bonus feature "paste a customer verbatim and see the impact on prediction" (sections 4.4/4.8) is not integrated into the interface.

The verbatim generation pipeline (`generate_verbatims.py`, `evaluate_text_signal.py`) is a separate ongoing workstream and is not yet connected to `app.py`.

## Remaining Project Tasks

- Run `generate_verbatims.py` (Gemini key required) and `evaluate_text_signal.py` to determine whether text signals provide additional value.
- Write the final report (3–6 pages) by assembling notes from sections 02 to 09.
- `README.md` has already been written separately.