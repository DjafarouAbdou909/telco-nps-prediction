import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import shap
import gradio as gr

train = pd.read_csv("data/processed/telco_nps_train.csv")
test = pd.read_csv("data/processed/telco_nps_test.csv")

X_train = train.drop(columns=["Customer ID", "NPS_Category"])
X_test = test.drop(columns=["Customer ID", "NPS_Category"])
y_train = train["NPS_Category"]
cat_cols = X_train.select_dtypes(include="str").columns.tolist()
num_cols = X_train.select_dtypes(exclude="str").columns.tolist()

prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols), ("num", StandardScaler(), num_cols)])
pipe_lr = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
pipe_lr.fit(X_train, y_train)

feat_names = pipe_lr.named_steps["prep"].get_feature_names_out()
X_train_enc = pipe_lr.named_steps["prep"].transform(X_train)
clf = pipe_lr.named_steps["clf"]
masker = shap.maskers.Independent(X_train_enc, max_samples=X_train_enc.shape[0])
explainer = shap.LinearExplainer(clf, masker, feature_names=feat_names)

FEATURE_COLS = X_train.columns.tolist()
CUSTOMER_CHOICES = test["Customer ID"].tolist()

def clean_feat(n):
    return n.replace("cat__", "").replace("num__", "")

def predict_from_row(row_df):
    proba = pipe_lr.predict_proba(row_df)[0]
    classes = pipe_lr.named_steps["clf"].classes_
    pred_class = classes[np.argmax(proba)]
    proba_dict = {str(k): float(v) for k, v in zip(classes, proba)}
    row_enc = pipe_lr.named_steps["prep"].transform(row_df)
    sv = explainer(row_enc).values[0]
    det_idx = list(classes).index("Detractor")
    sv_det = sv[:, det_idx]
    top_drivers = pd.Series(sv_det, index=[clean_feat(f) for f in feat_names]).sort_values(key=abs, ascending=False).head(5)
    return pred_class, proba_dict, top_drivers

def make_driver_plot(top_drivers, title):
    fig, ax = plt.subplots(figsize=(5,3))
    sorted_drivers = top_drivers.sort_values()
    colors = ["#B22222" if v > 0 else "#2E8A5C" for v in sorted_drivers.values]
    sorted_drivers.plot(kind="barh", ax=ax, color=colors)
    ax.set_title(title)
    ax.axvline(0, color="black", linewidth=0.7)
    plt.tight_layout()
    return fig

def predict_existing(customer_id):
    row = test[test["Customer ID"] == customer_id][FEATURE_COLS]
    if row.empty:
        return "Client introuvable", {}, None
    pred_class, proba_dict, top_drivers = predict_from_row(row)
    fig = make_driver_plot(top_drivers, "Top drivers (push toward Detractor)")
    return f"Prédiction : {pred_class}", proba_dict, fig

def predict_manual(gender, senior, partner, dependents, tenure, phone, multi_lines,
                    internet, security, backup, protection, tech_support, tv, movies,
                    contract, paperless, payment, monthly_charges, total_charges):
    # Reconstruction des features engineered à partir des inputs bruts
    n_services = sum([
        phone == "Yes", internet != "No",
        security == "Yes", backup == "Yes", protection == "Yes",
        tech_support == "Yes", tv == "Yes", movies == "Yes"
    ])
    charges_per_service = monthly_charges / n_services if n_services > 0 else monthly_charges
    household_size_proxy = int(partner == "Yes") + int(dependents == "Yes")
    is_autopay = int("automatic" in payment)

    row = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": int(senior), "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone, "MultipleLines": multi_lines,
        "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
        "DeviceProtection": protection, "TechSupport": tech_support, "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
        "n_services": n_services, "charges_per_service": charges_per_service,
        "household_size_proxy": household_size_proxy, "is_autopay": is_autopay
    }])[FEATURE_COLS]

    pred_class, proba_dict, top_drivers = predict_from_row(row)
    fig = make_driver_plot(top_drivers, "Top drivers for this profile")
    return f"Prédiction : {pred_class}", proba_dict, fig

with gr.Blocks(title="NPS Prediction : Retention Dashboard") as demo:
    gr.Markdown("# NPS Prediction :Outil de priorisation rétention")
    gr.Markdown(
        "Modèle : régression logistique (retenu en `05_evaluation` pour son lift sur la classe Détracteur). "
        "Les probabilités affichées ne sont pas parfaitement calibrées (voir `05_evaluation`) "
        "à utiliser pour **classer** les clients entre eux, pas comme un pourcentage de risque absolu."
    )
    with gr.Tab("Client existant"):
        customer_dropdown = gr.Dropdown(choices=CUSTOMER_CHOICES, label="Customer ID", value=CUSTOMER_CHOICES[0])
        predict_btn1 = gr.Button("Prédire", variant="primary")
        pred_output1 = gr.Textbox(label="Résultat")
        proba_output1 = gr.Label(label="Probabilités par catégorie")
        drivers_plot1 = gr.Plot(label="Principaux facteurs (SHAP)")
        predict_btn1.click(fn=predict_existing, inputs=customer_dropdown, outputs=[pred_output1, proba_output1, drivers_plot1])

    with gr.Tab("Saisie manuelle"):
        with gr.Row():
            gender_in = gr.Radio(["Male","Female"], label="Genre", value="Female")
            senior_in = gr.Radio([0,1], label="Senior Citizen", value=0)
            partner_in = gr.Radio(["Yes","No"], label="Partner", value="No")
            dependents_in = gr.Radio(["Yes","No"], label="Dependents", value="No")
        with gr.Row():
            tenure_in = gr.Slider(0, 72, value=12, label="Tenure (mois)")
            monthly_in = gr.Slider(18, 120, value=70, label="Monthly Charges ($)")
            total_in = gr.Number(value=840, label="Total Charges ($)")
        with gr.Row():
            contract_in = gr.Dropdown(["Month-to-month","One year","Two year"], label="Contract", value="Month-to-month")
            payment_in = gr.Dropdown(["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"], label="Payment Method", value="Electronic check")
            paperless_in = gr.Radio(["Yes","No"], label="Paperless Billing", value="Yes")
        with gr.Row():
            phone_in = gr.Radio(["Yes","No"], label="Phone Service", value="Yes")
            multilines_in = gr.Radio(["Yes","No","No phone service"], label="Multiple Lines", value="No")
            internet_in = gr.Dropdown(["DSL","Fiber optic","No"], label="Internet Service", value="Fiber optic")
        with gr.Row():
            security_in = gr.Radio(["Yes","No","No internet service"], label="Online Security", value="No")
            backup_in = gr.Radio(["Yes","No","No internet service"], label="Online Backup", value="No")
            protection_in = gr.Radio(["Yes","No","No internet service"], label="Device Protection", value="No")
        with gr.Row():
            tech_in = gr.Radio(["Yes","No","No internet service"], label="Tech Support", value="No")
            tv_in = gr.Radio(["Yes","No","No internet service"], label="Streaming TV", value="No")
            movies_in = gr.Radio(["Yes","No","No internet service"], label="Streaming Movies", value="No")

        predict_btn2 = gr.Button("Prédire", variant="primary")
        pred_output2 = gr.Textbox(label="Résultat")
        proba_output2 = gr.Label(label="Probabilités par catégorie")
        drivers_plot2 = gr.Plot(label="Principaux facteurs (SHAP)")

        predict_btn2.click(
            fn=predict_manual,
            inputs=[gender_in, senior_in, partner_in, dependents_in, tenure_in, phone_in, multilines_in,
                    internet_in, security_in, backup_in, protection_in, tech_in, tv_in, movies_in,
                    contract_in, paperless_in, payment_in, monthly_in, total_in],
            outputs=[pred_output2, proba_output2, drivers_plot2]
        )

if __name__ == "__main__":
    demo.launch()