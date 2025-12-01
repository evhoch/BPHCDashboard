# data_utils.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_historical_data(file, shelter_name: str) -> pd.DataFrame | None:
    if file is not None:
        df = pd.read_csv(file)

        # find date col
        date_col_candidates = [c for c in df.columns if c.lower() in ["date", "day"]]
        if not date_col_candidates:
            return None
        date_col = date_col_candidates[0]
        df[date_col] = pd.to_datetime(df[date_col])

        # find shelter guests col
        guests_col_candidates = [c for c in df.columns if "shelter guests" in c.lower()]
        if not guests_col_candidates:
            return None
        guests_col = guests_col_candidates[0]

        df = df[[date_col, guests_col]].rename(
            columns={date_col: "date", guests_col: "Shelter Guests"}
        ).sort_values("date").reset_index(drop=True)

    else:
        # simulate ~6 months
        end_date = datetime.today().date()
        start_date = end_date - pd.Timedelta(days=180)
        dates = pd.date_range(start=start_date, end=end_date, freq="D")

        base = 250 + 20 * np.sin(np.linspace(0, 4 * np.pi, len(dates)))
        noise = np.random.normal(scale=8, size=len(dates))
        sg = base + noise

        df = pd.DataFrame({"date": dates, "Shelter Guests": sg})

    if "women" in shelter_name.lower():
        df["Shelter Guests"] = df["Shelter Guests"] - 25

    return df
