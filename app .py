
import gradio as gr
import joblib
import pandas as pd
import numpy as np

# Load components
try:
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    raise FileNotFoundError("scaler.pkl not found. Ensure the file is included in the Space.")
try:
    selected_features = joblib.load('selected_features.pkl')
except FileNotFoundError:
    raise FileNotFoundError("selected_features.pkl not found. Ensure the file is included in the Space.")
try:
    results_df = pd.read_excel('results_df.xlsx')
except FileNotFoundError:
    results_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost', 'Neural Network (MLP)'],
        'Accuracy': [0.0, 0.0, 0.0, 0.0],
        'Note': ['Placeholder data - replace with actual metrics'] * 4
    })
try:
    df = pd.read_csv('data_rename.csv')
except FileNotFoundError:
    raise FileNotFoundError("data_rename.csv not found. Ensure the file is included in the Space.")

# Correct feature names by removing leading/trailing spaces
selected_features = [feature.strip() for feature in selected_features]
# Check scaler's expected feature names
scaler_feature_names = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else None
if scaler_feature_names is not None:
    if ' op_profit_rate' in scaler_feature_names and 'op_profit_rate' in selected_features:
        selected_features = [' op_profit_rate' if feature == 'op_profit_rate' else feature for feature in selected_features]
print("Corrected Selected Features:", selected_features)
print("Number of Features:", len(selected_features))

model_files = {
    "Logistic Regression": "Logistic_Regression.pkl",
    "Random Forest": "Random_Forest.pkl",
    "XGBoost": "XGBoost.pkl",
    "Neural Network (MLP)": "Neural_Network_(MLP).pkl"
}
models = {}
for name, path in model_files.items():
    try:
        models[name] = joblib.load(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"{path} not found. Ensure the file is included in the Space.")

# Map raw feature names to clean labels with emojis
feature_label_map = {
    'roa_c': 'Return on Assets (ROA) 📈',
    'op_gross_margin': 'Operating Gross Margin 💰',
    ' op_profit_rate': 'Operating Profit Rate 🏢',
    'nonindustry_inc_ratio': 'Non-industry Income Ratio 📊',
    'cash_flow_rate': 'Cash Flow Rate 💵',
    'debt_interest_rate': 'Debt Interest Rate 📉',
    'tax_rate_a': 'Tax Rate A 🧾',
    'net_val_share_b': 'Net Value Per Share (B) 🏦',
    'cash_flow_per_share': 'Cash Flow Per Share 💳',
    'realized_gross_growth': 'Realized Gross Profit Growth 🚀',
    'op_profit_growth': 'Operating Profit Growth 📈',
    'aftertax_profit_growth': 'After-tax Profit Growth 🏛️',
    'net_value_growth': 'Net Value Growth 📈',
    'asset_return_growth': 'Asset Return Growth 🧮',
    'current_ratio': 'Current Ratio 📊',
    'interest_expense_ratio': 'Interest Expense Ratio 🏦',
    'debt_to_networth': 'Debt to Net Worth Ratio 🏚️',
    'borrowing_dependency': 'Borrowing Dependency 🔗',
    'inv_and_rec_to_netval': 'Inventory and Receivables to Net Value 🏢',
    'op_profit_per_person': 'Operating Profit Per Person 👷',
    'allocation_per_person': 'Allocation Per Person 📋',
    'workingcap_to_assets': 'Working Capital to Assets ⚙️',
    'cash_to_assets': 'Cash to Assets 💵🏢',
    'cash_to_liability': 'Cash to Liability 📉',
    'currliability_to_assets': 'Current Liability to Assets 🔥',
    'longtermliability_to_currassets': 'Long-term Liability to Current Assets 🏗️',
    'expense_to_assets': 'Expense to Assets Ratio 📚',
    'workingcap_turnover': 'Working Capital Turnover 🔄',
    'equity_to_longtermliability': 'Equity to Long-term Liability 🏛️',
    'no_credit_interval': 'No Credit Interval (days) ⏳'
}

# Build inputs: clean labels with emojis for the UI
model_dropdown = gr.Dropdown(choices=list(model_files.keys()), label="Select Model")
feature_inputs = [gr.Number(label=feature_label_map.get(feature, feature)) for feature in selected_features]

# Prediction function
def predict_bankruptcy(model_name, *inputs):
    try:
        if any(x is None for x in inputs):
            return "Error: All inputs must be provided."
        model = models[model_name]
        input_df = pd.DataFrame([inputs], columns=selected_features)
        full_input = pd.DataFrame(np.zeros((1, len(df.columns) - 1)), columns=df.drop('bankrupt', axis=1).columns)
        for feature in selected_features:
            if feature in full_input.columns:
                full_input[feature] = input_df[feature]
        input_scaled = scaler.transform(full_input)
        input_scaled_df = pd.DataFrame(input_scaled, columns=df.drop('bankrupt', axis=1).columns)
        input_selected = input_scaled_df[selected_features]
        prediction = model.predict(input_selected)[0]
        probability = model.predict_proba(input_selected)[0][1]
        result = "Bankrupt" if prediction == 1 else "Not Bankrupt"
        return f"{result} (Probability: {probability:.2%})"
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio Interface
with gr.Interface(
    fn=predict_bankruptcy,
    inputs=[model_dropdown] + feature_inputs,
    outputs="text",
    title="Bankruptcy Prediction System",
    description="Select a model and input financial ratios to predict bankruptcy."
) as interface:
    gr.Markdown("## Model Performance Metrics")
    gr.Dataframe(results_df)

    def load_sample_input():
        sample = df[selected_features].iloc[0].to_dict()
        return [sample[feature] for feature in selected_features]

    sample_button = gr.Button("Load Sample Input")
    sample_button.click(
        fn=load_sample_input,
        inputs=None,
        outputs=feature_inputs
    )

# Launch the Gradio interface
interface.launch()
