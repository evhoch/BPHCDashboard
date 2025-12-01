# model_pipeline.py
import numpy as np
import pandas as pd
from datetime import timedelta

def make_dummy_forecast(
    historical_df: pd.DataFrame,
    horizon_days: int,
    interval_width: float = 15.0,
) -> pd.DataFrame:
    """
    Given a historical dataframe with columns:
        - 'date' (datetime)
        - 'Shelter Guests' (numeric)
    create a dummy forecast for the next `horizon_days` days.

    Returns a dataframe with columns:
        - 'date'
        - 'Shelter Guests'
        - 'lower'
        - 'upper'
    """

    if historical_df.empty:
        raise ValueError("historical_df is empty in make_dummy_forecast")

    # Last observed date and value
    last_actual_date = historical_df["date"].max()
    last_actual_value = historical_df["Shelter Guests"].iloc[-1]

    # Future dates
    forecast_dates = pd.date_range(
        start=last_actual_date + pd.Timedelta(days=1),
        periods=horizon_days,
        freq="D",
    )

    # Simple dummy forecast: small drift + noise
    drift = np.linspace(0, 10, horizon_days)
    noise = np.random.normal(scale=5, size=horizon_days)
    forecast_values = last_actual_value + drift + noise

    lower_bounds = forecast_values - interval_width
    upper_bounds = forecast_values + interval_width

    forecast_df = pd.DataFrame(
        {
            "date": forecast_dates,
            "Shelter Guests": forecast_values,
            "lower": lower_bounds,
            "upper": upper_bounds,
        }
    )

    return forecast_df
