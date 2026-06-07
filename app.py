import streamlit as st
import pandas as pd
import numpy as np
import joblib                     # <-- changed from pickle
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ------------------------------
# Page configuration
# ------------------------------
st.set_page_config(page_title="Urea Price Forecast", layout="wide")

st.title("🌾 AI‑Based Agricultural Input Cost Forecasting")
st.subheader("Forecast cost of agricultural inputs to aid financial planning for farmers and agri‑businesses")

# ------------------------------
# 1. Load trained model & training data
# ------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")      # <-- joblib instead of pickle
    return model

@st.cache_data
def load_data():
    df = pd.read_csv("data/urea_data.csv")
    return df

try:
    model = load_model()
    df = load_data()
except Exception as e:
    st.warning(f"⚠️ Model or data not found. Using dummy values for demo. Error: {e}")
    # Dummy data for demonstration (you will replace)
    df = pd.DataFrame({
        "Year": [2020,2021,2022],
        "Month": [1,2,3],
        "UreaPrice": [1500,1600,1700],
        "NaturalGasPrice": [300,310,320],
        "CrudeOilPrice": [5000,5100,5200],
        "Urea_Lag1": [1480,1500,1600],
        "Urea_Lag3": [1450,1480,1500]
    })
    # Dummy model (linear regression) with 6 features
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(df[["NaturalGasPrice","CrudeOilPrice","Year","Month","Urea_Lag1","Urea_Lag3"]], df["UreaPrice"])

# ------------------------------
# 2. User Input Form (sidebar)
# ------------------------------
st.sidebar.header("📥 Input Parameters")

def user_input_features():
    natural_gas = st.sidebar.number_input("Natural Gas Price (₹)", value=320.0)
    crude_oil = st.sidebar.number_input("Crude Oil Price (₹)", value=5400.0)
    year = st.sidebar.selectbox("Year", options=list(range(2020, 2027)), index=6)  # default 2026
    month = st.sidebar.selectbox("Month", options=list(range(1,13)), index=5)       # June = 6
    lag1 = st.sidebar.number_input("Previous Month Urea Price (Lag1) (₹)", value=1800.0)
    lag3 = st.sidebar.number_input("Previous 3‑Month Urea Price (Lag3) (₹)", value=1750.0)
    return {
        "NaturalGasPrice": natural_gas,
        "CrudeOilPrice": crude_oil,
        "Year": year,
        "Month": month,
        "Urea_Lag1": lag1,
        "Urea_Lag3": lag3
    }

input_data = user_input_features()

# Predict button
if st.sidebar.button("🔮 Predict Urea Price"):
    st.session_state["predicted"] = True
else:
    if "predicted" not in st.session_state:
        st.session_state["predicted"] = False

# ------------------------------
# 3. Make Prediction (6 features, correct order)
# ------------------------------
features = np.array([[
    input_data["NaturalGasPrice"],
    input_data["CrudeOilPrice"],
    input_data["Year"],
    input_data["Month"],
    input_data["Urea_Lag1"],
    input_data["Urea_Lag3"]
]])
predicted_price = model.predict(features)[0]

# ------------------------------
# 4. Display Prediction Result Card
# ------------------------------
st.header("📊 Prediction Result")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Predicted Urea Price", f"₹{predicted_price:.2f}")
col2.metric("Natural Gas Price Used", f"₹{input_data['NaturalGasPrice']:.2f}")
col3.metric("Crude Oil Price Used", f"₹{input_data['CrudeOilPrice']:.2f}")
col4.metric("Forecast Month", f"{input_data['Year']}-{input_data['Month']:02d}")

# ------------------------------
# 5. Historical Trend Graph
# ------------------------------
st.header("📈 Historical Urea Price Trend")
df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01")
fig = px.line(df, x="Date", y="UreaPrice", title="Urea Price over Time",
              labels={"UreaPrice": "Price (₹)", "Date": "Year-Month"})
st.plotly_chart(fig, width='stretch')   # <-- changed from use_container_width

# ------------------------------
# 6. Feature Impact Section (updated to 6 features)
# ------------------------------
st.header("🔍 Feature Impact")
if hasattr(model, "coef_"):
    coef = model.coef_
    feature_names = ["Natural Gas Price", "Crude Oil Price", "Year", "Month", "Urea_Lag1", "Urea_Lag3"]
    importance = np.abs(coef) / np.sum(np.abs(coef)) * 100
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance (%)": importance})
elif hasattr(model, "feature_importances_"):
    importance = model.feature_importances_ * 100
    feature_names = ["Natural Gas Price", "Crude Oil Price", "Year", "Month", "Urea_Lag1", "Urea_Lag3"]
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance (%)": importance})
else:
    # Dummy example (6 features)
    importance_df = pd.DataFrame({
        "Feature": ["Natural Gas Price", "Crude Oil Price", "Year", "Month", "Urea_Lag1", "Urea_Lag3"],
        "Importance (%)": [35, 30, 10, 5, 15, 5]
    })
st.dataframe(importance_df, width='stretch')   # <-- changed

# Bar chart for feature impact
fig2 = px.bar(importance_df, x="Feature", y="Importance (%)", title="Feature Importance")
st.plotly_chart(fig2, width='stretch')         # <-- changed

# ------------------------------
# 7. Actual vs Predicted Chart (on training data)
# ------------------------------
st.header("📉 Actual vs Predicted Urea Price (Training)")
# Use the same 6 features for X_train
X_train = df[["NaturalGasPrice", "CrudeOilPrice", "Year", "Month", "Urea_Lag1", "Urea_Lag3"]]
y_train = df["UreaPrice"]
y_pred_train = model.predict(X_train)

comparison_df = pd.DataFrame({
    "Date": df["Date"],
    "Actual": y_train,
    "Predicted": y_pred_train
})
fig3 = px.line(comparison_df, x="Date", y=["Actual", "Predicted"],
               title="Actual vs Predicted",
               labels={"value": "Price (₹)", "Date": "Time"})
st.plotly_chart(fig3, width='stretch')         # <-- changed

# ------------------------------
# 8. Model Performance Metrics
# ------------------------------
r2 = r2_score(y_train, y_pred_train)
rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
mae = mean_absolute_error(y_train, y_pred_train)

st.header("📐 Model Performance")
metrics_df = pd.DataFrame({
    "Metric": ["R² Score", "RMSE (₹)", "MAE (₹)"],
    "Value": [f"{r2:.4f}", f"{rmse:.2f}", f"{mae:.2f}"]
})
st.table(metrics_df)

# ------------------------------
# 9. Download Options
# ------------------------------
st.header("📎 Download Predictions")

if st.session_state["predicted"]:
    user_pred_df = pd.DataFrame([{
        "NaturalGasPrice": input_data["NaturalGasPrice"],
        "CrudeOilPrice": input_data["CrudeOilPrice"],
        "Year": input_data["Year"],
        "Month": input_data["Month"],
        "Urea_Lag1": input_data["Urea_Lag1"],
        "Urea_Lag3": input_data["Urea_Lag3"],
        "Predicted_UreaPrice_₹": predicted_price
    }])
    csv = user_pred_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"urea_prediction_{input_data['Year']}_{input_data['Month']}.csv",
        mime="text/csv"
    )

# PDF report via HTML
st.info("💡 To download a PDF report: click the **Generate Report** button below and then 'Save as PDF'.")

if st.button("🖨️ Generate Report (for PDF)"):
    report_html = f"""
    <html>
    <head><title>Urea Price Forecast Report</title></head>
    <body>
    <h1>Urea Price Forecast Report</h1>
    <p><b>Predicted Urea Price:</b> ₹{predicted_price:.2f}</p>
    <p><b>Inputs:</b> Natural Gas = ₹{input_data['NaturalGasPrice']}, Crude Oil = ₹{input_data['CrudeOilPrice']}</p>
    <p><b>Forecast Month:</b> {input_data['Year']}-{input_data['Month']}</p>
    <h3>Model Performance</h3>
    {metrics_df.to_html()}
    <h3>Feature Importance</h3>
    {importance_df.to_html()}
    </body>
    </html>
    """
    st.download_button(
        label="📄 Download HTML Report (then print to PDF)",
        data=report_html,
        file_name="urea_forecast_report.html",
        mime="text/html"
    )