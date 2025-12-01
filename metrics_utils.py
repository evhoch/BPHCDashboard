# metrics_utils.py
import pandas as pd

def compute_forecast_summary(historical_df: pd.DataFrame, forecast_df: pd.DataFrame):
    max_idx = forecast_df["Shelter Guests"].idxmax()
    max_sg = forecast_df.loc[max_idx, "Shelter Guests"]
    max_sg_date = forecast_df.loc[max_idx, "date"].date()

    mean_forecast = forecast_df["Shelter Guests"].mean()
    min_forecast = forecast_df["Shelter Guests"].min()

    last_hist_date = historical_df["date"].iloc[-1].date()
    last_hist_value = historical_df["Shelter Guests"].iloc[-1]

    summary_data = {
        "Metric": [
            "Forecast horizon (days)",
            "Last actual Shelter Guests date",
            "Last actual Shelter Guests",
            "Mean forecasted Shelter Guests",
            "Min forecasted Shelter Guests",
            "Max forecasted Shelter Guests",
            "Date of max forecast",
        ],
        "Value": [
            None,  # fill in horizon in app.py
            last_hist_date,
            f"{last_hist_value:.1f}",
            f"{mean_forecast:.1f}",
            f"{min_forecast:.1f}",
            f"{max_sg:.1f}",
            str(max_sg_date),
        ],
    }
    return pd.DataFrame(summary_data)
