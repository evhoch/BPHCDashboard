import streamlit as st
import pandas as pd

from data_utils import load_historical_data
# from model_pipeline import make_dummy_forecast  # <- no longer used
from charts import build_forecast_figure
from metrics_utils import compute_forecast_summary

from finalBPHCModel import make_shelter_forecast


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

# ---------- Load or simulate historical data ----------
# ---------- Load historical data from uploaded CSV ----------
if uploaded_file is not None:
    # Read the raw BPHC CSV
    df_raw = pd.read_csv(uploaded_file)

    # Make sure Date is datetime
    df_raw["Date"] = pd.to_datetime(df_raw["Date"])

    # Build a simple historical_df for plotting/metrics
    # (using Census_Men as "Shelter Guests" for now)
    historical_df = df_raw[["Date", "Census_Men"]].rename(
        columns={"Date": "date", "Census_Men": "Shelter Guests"}
    )

    historical_df = df_raw[["Date", "Census_Men"]].rename(
    columns={"Date": "date", "Census_Men": "Shelter Guests"}
)

    # Use last 60 days to estimate variability
    hist_recent = historical_df.tail(60)
    sigma = hist_recent["Shelter Guests"].std()


    if not historical_df.empty:
        # Restrict to last ~2 months (60 days) for plotting
        max_date = historical_df["date"].max()
        three_months_ago = max_date - pd.Timedelta(days=60)
        hist_plot_df = historical_df[historical_df["date"] >= three_months_ago].copy()

    # ---------- Forecast generation (via BPHC model) ----------
        model_forecast_df = make_shelter_forecast(
            df_raw,
            horizon_days=forecast_horizon_days,
            target_col="Census_Men",
        )

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

        # --- Add trivial prediction bands so charts.py can compute shading ---
        z = 1.0  # or 1.96 for ~95% if you want fatter bands

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

        fig = build_forecast_figure(plot_df, shelter_option)
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

        st.header("Model & Diagnostics")
        st.caption("Details about the model and summary diagnostics based on recent history and the forecast horizon.")


        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Model Info")
            st.write(f"**Selected shelter:** {shelter_option}")
            st.write("**Model:** BPHC RF + XGBoost ensemble")
            st.write("**Training status:** Loaded from integrated script")

        with col2:
            st.subheader("Diagnostics")

            # Use last 60 days of history for basic stats
            if len(historical_df) >= 1:
                recent_hist = historical_df.tail(60)

                hist_mean = recent_hist["Shelter Guests"].mean()
                hist_std  = recent_hist["Shelter Guests"].std()
                last_actual = historical_df["Shelter Guests"].iloc[-1]
            else:
                hist_mean = float("nan")
                hist_std = float("nan")
                last_actual = float("nan")

            # Forecast stats over the selected horizon
            if len(forecast_df) >= 1:
                fc_mean = forecast_df["Shelter Guests"].mean()
                fc_min  = forecast_df["Shelter Guests"].min()
                fc_max  = forecast_df["Shelter Guests"].max()
            else:
                fc_mean = fc_min = fc_max = float("nan")

            diag_df = pd.DataFrame({
                "Metric": [
                    "Last actual census",
                    "Historical mean (last 60 days)",
                    "Historical std (last 60 days)",
                    f"Forecast mean (next {forecast_horizon_days} days)",
                    "Forecast min (next horizon)",
                    "Forecast max (next horizon)",
                ],
                "Value": [
                    f"{last_actual:.1f}",
                    f"{hist_mean:.1f}",
                    f"{hist_std:.1f}",
                    f"{fc_mean:.1f}",
                    f"{fc_min:.1f}",
                    f"{fc_max:.1f}",
                ],
            })

            st.table(diag_df)



    else:
        st.warning("Historical data is empty after processing. Check your CSV.")
else:
    st.warning("No historical data available. Please upload a valid BPHC CSV.")
