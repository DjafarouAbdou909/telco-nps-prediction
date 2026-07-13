import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr

from src.model import train_production_model
from src.explain import build_explainer, top_drivers_for_row
from src.features import reconstruct_manual_row

train = pd.read_csv("data/processed/telco_nps_train.csv")
test = pd.read_csv("data/processed/telco_nps_test.csv")

X_train = train.drop(columns=["Customer ID", "NPS_Category"])
y_train = train["NPS_Category"]

pipe = train_production_model(X_train, y_train)
explainer, feat_names = build_explainer(pipe, X_train)

FEATURE_COLS = X_train.columns.tolist()
CUSTOMER_CHOICES = test["Customer ID"].tolist()


def make_driver_plot(top_drivers, title):
    fig, ax = plt.subplots(figsize=(5, 3))
    sorted_drivers = top_drivers.sort_values()
    colors = ["#B22222" if v > 0 else "#2E8A5C" for v in sorted_drivers.values]
    sorted_drivers.plot(kind="barh", ax=ax, color=colors)
    ax.set_title(title)
    ax.axvline(0, color="black", linewidth=0.7)
    plt.tight_layout()
    return fig


def run_prediction(row_df, plot_title):
    proba = pipe.predict_proba(row_df)[0]
    classes = pipe.named_steps["clf"].classes_
    pred_class = classes[proba.argmax()]
    proba_dict = {str(k): float(v) for k, v in zip(classes, proba)}
    top_drivers = top_drivers_for_row(pipe, explainer, feat_names, row_df, target_class="Detractor")
    fig = make_driver_plot(top_drivers, plot_title)
    return f"Prediction: {pred_class}", proba_dict, fig


def predict_existing(customer_id):
    row = test[test["Customer ID"] == customer_id][FEATURE_COLS]
    if row.empty:
        return "Customer not found", {}, None
    return run_prediction(row, "Top drivers (push toward Detractor)")


def predict_manual(gender, senior, partner, dependents, tenure, phone, multi_lines,
                    internet, security, backup, protection, tech_support, tv, movies,
                    contract, paperless, payment, monthly_charges, total_charges):
    raw_inputs = {
        "gender": gender, "SeniorCitizen": int(senior), "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone, "MultipleLines": multi_lines,
        "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
        "DeviceProtection": protection, "TechSupport": tech_support, "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
    }
    row = reconstruct_manual_row(FEATURE_COLS, **raw_inputs)
    return run_prediction(row, "Top drivers for this profile")


with gr.Blocks(title="NPS Prediction - Retention Dashboard") as demo:
    gr.Markdown("# NPS Prediction - Retention Prioritization Tool")
    gr.Markdown(
        "Model: logistic regression (selected in `05_evaluation` for its lift on the Detractor class). "
        "The probabilities shown are not perfectly calibrated (see `05_evaluation`) "
        "use them to **rank** customers relative to each other, not as an absolute risk percentage."
    )
    with gr.Tab("Existing Customer"):
        customer_dropdown = gr.Dropdown(choices=CUSTOMER_CHOICES, label="Customer ID", value=CUSTOMER_CHOICES[0])
        predict_btn1 = gr.Button("Predict", variant="primary")
        pred_output1 = gr.Textbox(label="Result")
        proba_output1 = gr.Label(label="Probability by Category")
        drivers_plot1 = gr.Plot(label="Top Drivers (SHAP)")
        predict_btn1.click(fn=predict_existing, inputs=customer_dropdown, outputs=[pred_output1, proba_output1, drivers_plot1])

    with gr.Tab("Manual Entry"):
        with gr.Row():
            gender_in = gr.Radio(["Male", "Female"], label="Gender", value="Female")
            senior_in = gr.Radio([0, 1], label="Senior Citizen", value=0)
            partner_in = gr.Radio(["Yes", "No"], label="Partner", value="No")
            dependents_in = gr.Radio(["Yes", "No"], label="Dependents", value="No")
        with gr.Row():
            tenure_in = gr.Slider(0, 72, value=12, label="Tenure (months)")
            monthly_in = gr.Slider(18, 120, value=70, label="Monthly Charges ($)")
            total_in = gr.Number(value=840, label="Total Charges ($)")
        with gr.Row():
            contract_in = gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month")
            payment_in = gr.Dropdown(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], label="Payment Method", value="Electronic check")
            paperless_in = gr.Radio(["Yes", "No"], label="Paperless Billing", value="Yes")
        with gr.Row():
            phone_in = gr.Radio(["Yes", "No"], label="Phone Service", value="Yes")
            multilines_in = gr.Radio(["Yes", "No", "No phone service"], label="Multiple Lines", value="No")
            internet_in = gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service", value="Fiber optic")
        with gr.Row():
            security_in = gr.Radio(["Yes", "No", "No internet service"], label="Online Security", value="No")
            backup_in = gr.Radio(["Yes", "No", "No internet service"], label="Online Backup", value="No")
            protection_in = gr.Radio(["Yes", "No", "No internet service"], label="Device Protection", value="No")
        with gr.Row():
            tech_in = gr.Radio(["Yes", "No", "No internet service"], label="Tech Support", value="No")
            tv_in = gr.Radio(["Yes", "No", "No internet service"], label="Streaming TV", value="No")
            movies_in = gr.Radio(["Yes", "No", "No internet service"], label="Streaming Movies", value="No")

        predict_btn2 = gr.Button("Predict", variant="primary")
        pred_output2 = gr.Textbox(label="Result")
        proba_output2 = gr.Label(label="Probability by Category")
        drivers_plot2 = gr.Plot(label="Top Drivers (SHAP)")

        predict_btn2.click(
            fn=predict_manual,
            inputs=[gender_in, senior_in, partner_in, dependents_in, tenure_in, phone_in, multilines_in,
                    internet_in, security_in, backup_in, protection_in, tech_in, tv_in, movies_in,
                    contract_in, paperless_in, payment_in, monthly_in, total_in],
            outputs=[pred_output2, proba_output2, drivers_plot2]
        )

if __name__ == "__main__":
    demo.launch()