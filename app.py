import streamlit as st
import pandas as pd
import time 

from data_utils import load_historical_data
from charts import build_forecast_figure
from metrics_utils import compute_forecast_summary

from Jiao_DL_Linear_ import make_dl_forecast


# ---------- Page config ----------
st.set_page_config(
    page_title="Shelter Guests Forecast",
    layout="wide"
)

st.title("Shelter Guests Forecast Dashboard")

# ---------- Sidebar: Controls ----------
st.sidebar.header("Controls")

# Shelter selection
shelter_option = st.sidebar.selectbox(
    "Select shelter",
    ["112 Southampton (Men's Shelter)", "Woods Mullen (Women's shelter)"]
)

if "Men" in shelter_option:
    target_col = "Census_Men"
elif "Women" in shelter_option:
    target_col = "Census_Women"
else:
    st.error("Unknown shelter option")
    st.stop()

# Forecast horizon in days
forecast_horizon_days = st.sidebar.slider(
    "Forecast horizon (days)",
    min_value=7,
    max_value=35,
    value=14,
    step=7
)

# CSV upload
uploaded_file = st.sidebar.file_uploader(
    "Upload historical shelter guests CSV",
    type=["csv"],
    help="For now, expects columns named 'Date' and BPHC census columns "
         "(e.g. Census_Men, Census_Women, etc.)."
)

# ---------- Load historical data from uploaded CSV ----------
if uploaded_file is not None:
    # Read the raw BPHC CSV
    df_raw = pd.read_csv(uploaded_file)

    # Make sure Date is datetime
    df_raw["Date"] = pd.to_datetime(df_raw["Date"])

    # Build historical_df for plotting/metrics using the selected target_col
    # target_col is "Census_Men" or "Census_Women"
    historical_df = df_raw[["Date", target_col]].dropna().rename(
        columns={"Date": "date", target_col: "Shelter Guests"}
    )
    
    # Optional: Sort by date just to be safe
    historical_df = historical_df.sort_values("date")

    # Use last 60 days to estimate variability
    hist_recent = historical_df.tail(60)
    sigma = hist_recent["Shelter Guests"].std()


    if not historical_df.empty:
        # Restrict to last ~2 months (60 days) for plotting
        max_date = historical_df["date"].max()
        three_months_ago = max_date - pd.Timedelta(days=60)
        hist_plot_df = historical_df[historical_df["date"] >= three_months_ago].copy()

            
        # ---------- Forecast generation (via BPHC model) ----------
        confidence_level=0.80

        start_time = time.time()

        model_forecast_df, model_metrics = make_dl_forecast(
            df_raw,
            horizon_days=forecast_horizon_days,
            target_col=target_col,
            lookback=30,  # You can change this lookback window if needed
            confidence_level=confidence_level
        )
        end_time = time.time()
        training_duration = end_time - start_time

        # Start from whatever the model gave us
        forecast_df = model_forecast_df.copy()


        # --- Normalize the date column name ---
        if "date" in forecast_df.columns:
            # already good
            pass
        elif "Date" in forecast_df.columns:
            forecast_df = forecast_df.rename(columns={"Date": "date"})
        else:
            st.error("Model output is missing a date column.")
            st.write("Columns:", list(forecast_df.columns))
            st.stop()

        # --- Normalize the prediction column name to 'Shelter Guests' ---
        if "Shelter Guests" in forecast_df.columns:
            # already good
            pass
        elif "Predicted" in forecast_df.columns:
            forecast_df = forecast_df.rename(columns={"Predicted": "Shelter Guests"})
        elif "forecast" in forecast_df.columns:
            forecast_df = forecast_df.rename(columns={"forecast": "Shelter Guests"})
        else:
            st.error("Model output is missing a prediction column.")
            st.write("Columns:", list(forecast_df.columns))
            st.stop()

        # --- Attach prediction bands ---

        # Case 1: bootstrap intervals are present from the model
        if "Predicted_lower" in forecast_df.columns and "Predicted_upper" in forecast_df.columns:
            # Rename to the standard names charts.py expects
            forecast_df = forecast_df.rename(
                columns={
                    "Predicted_lower": "lower",
                    "Predicted_upper": "upper",
                }
            )
            # Ensure no negative guests
            forecast_df["lower"] = forecast_df["lower"].clip(lower=0)

        # Case 2: no bootstrap columns -> fall back to simple sigma-based band
        else:
            print("No confidence level, using sigma!")
            # z can be 1.96 for ~95% if you want
            z = 1.0

            y_hat = forecast_df["Shelter Guests"]
            forecast_df["lower"] = (y_hat - z * sigma).clip(lower=0)
            forecast_df["upper"] = y_hat + z * sigma



        # Tag historical vs forecast for plotting
        hist_plot_df["type"] = "Historical"
        forecast_df["type"] = "Forecast"

        plot_df = pd.concat([hist_plot_df, forecast_df], ignore_index=True)
        plot_df["shelter"] = shelter_option

        # ---------- Main panel: Forecast & Historical Visualization ----------
        st.header(f"Forecast & Historical Shelter Guests — {shelter_option}")

        fig = build_forecast_figure(plot_df, shelter_option, target_col)

        fig.update_yaxes(range=[0, None])

        st.plotly_chart(fig, width="stretch")

        # ---------- Summary metrics section ----------
        st.subheader("Forecast Summary")

        summary_df = compute_forecast_summary(historical_df, forecast_df)
        summary_df.loc[
            summary_df["Metric"] == "Forecast horizon (days)", "Value"
        ] = str(forecast_horizon_days)
        summary_df["Value"] = summary_df["Value"].astype(str)
        st.table(summary_df)

        # ---------- Model & Diagnostics section ----------
        st.header("Model & Diagnostics")
        st.caption("Details about the model and summary diagnostics based on recent history and the forecast horizon.")


        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Model Info")
            st.write(f"**Selected shelter:** {shelter_option}")
            st.write("**Model:** DL Linear (PyTorch)")
            st.write(f"**Training time:** {training_duration:.2f} seconds") 
            st.write("**Training status:** Retrained live on uploaded data")

        with col2:
            st.subheader("Diagnostics")

            # Extract metrics from the model_metrics dictionary we returned
            test_rmse = model_metrics.get("test_RMSE", 0)
            test_mae  = model_metrics.get("test_MAE", 0)
            conf_lvl  = model_metrics.get("confidence_level", 0.90) * 100
            
            # Calculate "Average Margin of Error" from the forecast dataframe itself
            # (Upper Band - Prediction) averaged over the horizon
            if "upper" in forecast_df.columns and "Shelter Guests" in forecast_df.columns:
                avg_width = (forecast_df["upper"] - forecast_df["Shelter Guests"]).mean()
            else:
                avg_width = 0

            diag_df = pd.DataFrame({
                "Metric": [
                    "Test Set RMSE", 
                    "Test Set MAE", 
                    "Confidence Level",
                    "Avg. Margin of Error (+/-)"
                ],
                "Value": [
                    f"{test_rmse:.2f}",
                    f"{test_mae:.2f}",
                    f"{conf_lvl:.0f}%", 
                    f"{avg_width:.1f} guests"
                ],
                "Description": [
                    "Root Mean Squared Error on unseen test data",
                    "Average absolute error on unseen test data",
                    "Probability that true value is within bands",
                    "Average width of the error band for this forecast"
                ]
            })

            st.table(diag_df)



    else:
        st.warning("Historical data is empty after processing. Check your CSV.")
else:
    st.warning("No historical data available. Please upload a valid BPHC CSV.")
