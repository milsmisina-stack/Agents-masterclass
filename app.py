import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from prophet import Prophet

# =========================
# 🔐 Load API Key
# =========================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(
    page_title="Revenue Forecasting Agent (Prophet)",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Revenue Forecasting Agent (Prophet)")
st.caption("Upload an Excel file with **Date** and **Revenue** columns to generate a Prophet-based forecast and AI commentary.")

# Validate API key
if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set GROQ_API_KEY in a `.env` file or Streamlit Secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# =========================
# 📁 File Upload
# =========================
uploaded_file = st.file_uploader(
    "Upload your Excel or CSV file (must contain columns: Date, Revenue)",
    type=["xlsx", "xls", "csv"],
)

if uploaded_file is None:
    st.info("👆 Upload an Excel/CSV file to get started.")
    st.stop()

# =========================
# 🧹 Data Loading & Cleaning
# =========================
try:
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        raw_df = pd.read_csv(uploaded_file)
    else:
        raw_df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"❌ Error reading file: {e}")
    st.stop()

st.subheader("📄 Raw Data Preview")
st.dataframe(raw_df.head())

# Normalize column names
raw_df.columns = [c.strip().lower() for c in raw_df.columns]

if "date" not in raw_df.columns or "revenue" not in raw_df.columns:
    st.error("❌ The file must contain 'Date' and 'Revenue' columns (case-insensitive).")
    st.stop()

df = raw_df[["date", "revenue"]].copy()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")

df = df.dropna(subset=["date", "revenue"])
df = df.sort_values("date")

if df.empty or len(df) < 10:
    st.error("❌ Not enough valid rows to train a model. Need at least 10 rows with valid Date and Revenue.")
    st.stop()

st.subheader("📊 Cleaned Time Series Data")
st.dataframe(df.tail(20))

# =========================
# ⚙️ Sidebar: Forecast Settings
# =========================
with st.sidebar:
    st.header("⚙️ Forecast Settings")
    horizon_days = st.slider(
        "Forecast horizon (days)",
        min_value=7,
        max_value=365,
        step=7,
        value=90,
        help="How many days into the future to forecast.",
    )
    interval_width = st.slider(
        "Confidence interval",
        min_value=0.5,
        max_value=0.99,
        value=0.95,
        step=0.01,
        help="Width of the uncertainty interval for the forecast.",
    )
    daily_seasonality = st.checkbox("Daily seasonality", value=True)
    weekly_seasonality = st.checkbox("Weekly seasonality", value=True)
    yearly_seasonality = st.checkbox("Yearly seasonality", value=True)

# =========================
# 🔮 Prophet Forecast
# =========================
st.subheader("🔮 Prophet Revenue Forecast")

prophet_df = df.rename(columns={"date": "ds", "revenue": "y"})

with st.spinner("Training Prophet model and generating forecast..."):
    model = Prophet(
        interval_width=interval_width,
        daily_seasonality=daily_seasonality,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
    )
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=horizon_days)
    forecast = model.predict(future)

# Plot forecast
fig_forecast = model.plot(forecast)
plt.title("Revenue Forecast", fontsize=14)
st.pyplot(fig_forecast)

# Plot components
fig_components = model.plot_components(forecast)
st.pyplot(fig_components)

# =========================
# 📅 Forecast Table & Download
# =========================
forecast_tail = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon_days)
forecast_tail = forecast_tail.rename(
    columns={
        "ds": "date",
        "yhat": "forecast_revenue",
        "yhat_lower": "lower_bound",
        "yhat_upper": "upper_bound",
    }
)

st.subheader("📅 Forecast Table (Future Periods)")
st.dataframe(forecast_tail)

csv_data = forecast_tail.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Forecast as CSV",
    data=csv_data,
    file_name="revenue_forecast.csv",
    mime="text/csv",
)

# =========================
# 🤖 AI-Generated Forecast Commentary
# =========================
st.subheader("🤖 AI-Generated Forecast Commentary")

st.markdown(
    """
Click the button below to generate CFO-ready commentary on:
- Historical revenue trends and seasonality  
- Insights from the Prophet forecast  
- Risks and uncertainties (based on confidence intervals)  
- Actionable recommendations  
"""
)

if st.button("Generate AI Commentary"):
    with st.spinner("Calling Groq AI to generate commentary..."):
        # Compact data for the model (last 365 days + future)
        historical_sample = prophet_df.tail(365).to_dict(orient="records")
        forecast_sample = forecast_tail.to_dict(orient="records")

        data_for_ai = {
            "historical_data": historical_sample,
            "forecast_data": forecast_sample,
        }

        prompt = f"""
You are the Head of FP&A at a recurring-revenue business. Your task is to analyze the historical revenue and the Prophet-based forecast and provide:

1. A clear explanation of recent revenue trends and any seasonality you see.
2. Key insights from the forecast (growth, decline, stability, turning points).
3. Risks and uncertainties to watch, using the confidence intervals.
4. A concise, CFO-ready summary using the Pyramid Principle (start with the main message, then supporting points).
5. 3–5 actionable recommendations to improve revenue performance or forecast accuracy.

Here is the data (JSON):
{json.dumps(data_for_ai)}
"""

        try:
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a financial planning and analysis (FP&A) expert with deep "
                            "experience in revenue forecasting, time-series analysis, and SaaS/recurring business models."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model="llama3-8b-8192",
            )

            ai_commentary = response.choices[0].message.content

            st.markdown('<div class="analysis-container">', unsafe_allow_html=True)
            st.markdown(ai_commentary)
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error generating AI commentary: {e}")
