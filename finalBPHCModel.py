#!/usr/bin/env python
# coding: utf-8

# # BPHC Final Model Script: Random Forest and XG Boost Ensemble Method
# ### MIT Analytics Lab: Evan Hoch, Jacob Lebovitz, Jeremie Tarato, Jiao Zhao

# Import necessart packages for modeling, plotting, and error metrics

# In[5]:


# Dataframe
import pandas as pd 
import numpy as np 

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling Random Forest and sklearn packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

# Import XGBoost regressor
from xgboost import XGBRegressor

# If you get an error like "module [X] not defined" make sure you have installed that package in your current environment
# pip install [X] in your terminal usually fixes this error





def clean_with_neighbor_mean(
    df,
    features,
    diff_threshold=200,
    neighbor_window=2,   # number of neighbors on each side to use in the mean
    date_col="Date",
    inplace=False
):
    """
    Clean spikes in specified features using neighbor values.

    A cell df.iloc[i, feature] is considered an outlier if:
      1) abs(value - previous) > diff_threshold
      2) abs(value - next) > diff_threshold
      3) abs(previous - next) <= diff_threshold  (neighbors agree with each other)

    When an outlier is found, its value is replaced by the mean of the
    neighbor_window nearest valid values before and after that index
    (up to 2 before and 2 after by default).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe, assumed sorted by date and with a stable index.
    features : list of str
        Column names to clean.
    diff_threshold : float
        Threshold for detecting jumps between neighbors.
    neighbor_window : int
        How many neighbors on each side to average over.
    date_col : str
        Name of the date column, used only for interpretation and optional checks.
    inplace : bool
        If True, modify df in place. If False, work on a copy.

    Returns
    -------
    cleaned_df : pd.DataFrame
        Dataframe after cleaning.
    changes_log : pd.DataFrame
        Log of all changes with columns:
        ['index', 'feature', 'old_value', 'new_value']
    """
    if not inplace:
        df = df.copy()

    n_rows = len(df)
    changes = []

    # Step 0: Replace zeros in selected features using neighbor means BEFORE spike cleaning
    for feature in features:
        zero_idx = df.index[df[feature] == 0].tolist()

        for idx in zero_idx:
            pos = df.index.get_loc(idx)

            neighbor_vals = []
            for k in range(1, neighbor_window + 1):

                # before
                if pos - k >= 0:
                    v_before = df.iloc[pos - k][feature]
                    if not pd.isna(v_before) and v_before != 0:
                        neighbor_vals.append(v_before)

                # after
                if pos + k < n_rows:
                    v_after = df.iloc[pos + k][feature]
                    if not pd.isna(v_after) and v_after != 0:
                        neighbor_vals.append(v_after)

            if len(neighbor_vals) == 0:
                continue

            new_val = float(np.round(np.mean(neighbor_vals), 0))

            # Log the zero fix
            changes.append({
                "index": idx,
                "feature": feature,
                "old_value": 0,
                "new_value": new_val
            })

            # Apply the fix
            df.at[idx, feature] = new_val



    # Optional safety check, does not modify anything
    if date_col in df.columns and not df[date_col].is_monotonic_increasing:
        print(f"Warning: {date_col} is not strictly sorted. Make sure df is ordered by date before calling this function.")

    for feature in features:
        if feature not in df.columns:
            print(f"Feature '{feature}' not found in dataframe. Skipping.")
            continue

        series = df[feature]

        # Work with positional indices but keep track of original index labels
        for pos in range(1, n_rows - 1):
            current_val = series.iloc[pos]
            prev_val = series.iloc[pos - 1]
            next_val = series.iloc[pos + 1]

            # Skip if any of the three core values are missing
            if pd.isna(current_val) or pd.isna(prev_val) or pd.isna(next_val):
                continue

            # Check if the middle value looks like an isolated spike
            big_jump_prev = abs(current_val - prev_val) > diff_threshold
            big_jump_next = abs(current_val - next_val) > diff_threshold

            if not (big_jump_prev and big_jump_next):
                continue

            # Collect neighbor values around this position
            neighbor_vals = []
            for k in range(1, neighbor_window + 1):
                # before
                if pos - k >= 0:
                    val_before = series.iloc[pos - k]
                    if not pd.isna(val_before):
                        neighbor_vals.append(val_before)
                # after
                if pos + k < n_rows:
                    val_after = series.iloc[pos + k]
                    if not pd.isna(val_after):
                        neighbor_vals.append(val_after)

            if len(neighbor_vals) == 0:
                # nothing to average, skip
                continue

            new_val = float(np.round(np.mean(neighbor_vals), 0))

            # Only log and write if there is an actual change
            if new_val != current_val:
                idx_label = df.index[pos]
                changes.append(
                    {
                        "index": idx_label,
                        "feature": feature,
                        "old_value": current_val,
                        "new_value": new_val,
                    }
                )
                df.at[idx_label, feature] = new_val


    changes_log = pd.DataFrame(changes, columns=["index", "feature", "old_value", "new_value"])
    return df, changes_log


# In[116]:


# Normalize the data features so they all have a mean of 0 and SD of 1
# No need to normalize the target variable columns or Date/month/week
def normalize_df(
    df,
    exclude=("Census_Both", "Census_Men", "Census_Women",
             "TotalCapacityOtherShelters", "CensusLTS",
             "Date", "Month", "Week")
):
    """
    Normalize all numeric columns except those in `exclude`.
    """
    df_normalized = df.copy()
    cols = [c for c in df.columns if c not in exclude]
    means = df[cols].mean()
    stds = df[cols].std()

    df_normalized[cols] = (df[cols] - means) / stds

    return df_normalized



# In[117]:


# Create sin and cos columns for dates to capture seasonality
def transform_dates_in_sin(df):
    df = df.copy()  # avoid modifying original DataFrame

    month_number = df["Month"].dt.month
    w = df["Date"].dt.weekday + 1

    # compute all new columns first
    month_sin = np.sin(2 * np.pi * month_number / 12).round(2)
    month_cos = np.cos(2 * np.pi * month_number / 12).round(2)
    dow_sin = np.sin(2 * np.pi * w / 7).round(4)
    dow_cos = np.cos(2 * np.pi * w / 7).round(4)

    # combine everything at once
    df = pd.concat(
        [df.drop(columns=["Month"]),
         pd.DataFrame({
             "Month_sin": month_sin,
             "Month_cos": month_cos,
             "dow_sin": dow_sin,
             "dow_cos": dow_cos
         }, index=df.index)],
        axis=1
    )

    return df


# In[118]:


# Drop columns function
def dropping_features(df, features): 
    for feature in features: 
        df = df.drop(feature, axis =1)
    return df

# Drop capacity and census_both columns
#data_normalize = dropping_features(df_normalize, ["TotalCapacity_Men", "TotalCapacity_Women", "Census_Both", "TotalCapacity_Both"])


# In[119]:


# Drop columns with a high covariance (keep 1, drop other)

def drop_high_covariance_features(df, threshold=0.8, target="Census_Men"):
    data = df.copy()

    # Save date column separately
    date_series = pd.to_datetime(data["Date"])
    df = data.drop(columns="Date", axis=1)

    # Separate features and target
    features = [c for c in df.columns if c != target]

    # Compute correlations
    corr = df[features].corr()
    target_corr = df[features].corrwith(df[target])

    to_drop = set()
    counter = 0

    # Iterate over feature pairs
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            a = features[i]
            b = features[j]

            # Skip if already dropped
            if a in to_drop or b in to_drop:
                continue

            # Check correlation between features (must be scalar)
            corr_val = abs((corr.loc[a, b]))

            if corr_val > threshold:
                # Compare absolute correlation with target
                corr_a = abs((target_corr.get(a, 0.0)))
                corr_b = abs((target_corr.get(b, 0.0)))

                if corr_a >= corr_b:
                    to_drop.add(b)
                else:
                    to_drop.add(a)
                counter += 1

    print(f"Dropping {len(to_drop)} features due to high covariance: {to_drop}")

    # Drop and restore date column
    df_no_covariance = df.drop(columns=list(to_drop), errors="ignore")
    df_no_covariance.insert(0, "Date", date_series)

    return df_no_covariance




# In[120]:


def train_full_predict_future_tuned(clean_df, target_col="Census_Men", days_ahead=14):
    df = clean_df.copy().sort_values("Date").reset_index(drop=True)
    df[f"{target_col}_future"] = df[target_col].shift(-days_ahead)

    train_df = df.dropna(subset=[f"{target_col}_future"]).reset_index(drop=True)
    X = train_df.drop(columns=["Date", f"{target_col}_future"])
    y = train_df[f"{target_col}_future"]

    X = X.select_dtypes(include=[np.number])


    #imputer = SimpleImputer(strategy="mean")
    #X_imputed = imputer.fit_transform(X)

    #### XGB with GridSearchCV ####
    xgb = XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)
    param_grid = {
        "n_estimators": [100, 200, 500],
        "max_depth": [3, 5, 7]
    }
    grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)
    grid_search.fit(X, y)
    best_xgb = grid_search.best_estimator_

    #### RF with RandomizedSearchCV ####
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    param_dist = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_leaf': [5, 10]
    }
    tscv = TimeSeriesSplit(n_splits=5)
    rs = RandomizedSearchCV(rf, param_dist, n_iter=20, cv=tscv, scoring='neg_mean_squared_error',
                            random_state=42, n_jobs=-1, verbose=1)
    rs.fit(X, y)
    best_rf = rs.best_estimator_

    #### Predict future ####
    X_future = df.drop(columns=["Date", f"{target_col}_future"]).iloc[-days_ahead:]
    X_future_numeric = X_future.select_dtypes(include=[np.number])
    #X_future_imputed = imputer.transform(X_future_numeric)

    y_pred_xgb = best_xgb.predict(X_future_numeric)
    y_pred_rf = best_rf.predict(X_future_numeric)

    y_pred = 0.35*y_pred_xgb + 0.65*y_pred_rf
    last_dates = pd.to_datetime(df["Date"].iloc[-days_ahead:]) + pd.to_timedelta(days_ahead, unit="D")

    predictions_df = pd.DataFrame({"Date": last_dates, "Predicted": np.round(y_pred,0)}).reset_index(drop=True)
    return predictions_df


# In[121]:


def main(df, target_col="Census_Men", day_ahead=14, covariance_threshhold=0.8, exclude_from_normalization = ["Census_Both", "Census_Men", "Census_Women", "TotalCapacityOtherShelters", "CensusLTS", "Date", "Month", "Week"]): 
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = pd.to_datetime(df["Month"])



    features_to_clean = [
    "Census_Men",
    "Census_Women",
    "Census_Both",
    "TotalCapacityOtherShelters",
    ]

    data_clean, log_changes = clean_with_neighbor_mean(
        df,
        features=features_to_clean,
        diff_threshold=100,
        neighbor_window=2,
        date_col="Date",
        inplace=False
    )


    df_normalize = normalize_df(data_clean, exclude_from_normalization)
    data_normalize_sin = transform_dates_in_sin(df_normalize)
    features = ["TotalCapacity_Men", "TotalCapacity_Women", "Census_Both", "TotalCapacity_Both"]
    df_dropped = dropping_features(data_normalize_sin, features)
    df_dropped = df_dropped.drop(columns = "Week")

    df_no_cov = drop_high_covariance_features(df_dropped, threshold=covariance_threshhold)
    predictions = train_full_predict_future_tuned(df_no_cov, target_col=target_col, days_ahead=day_ahead)
    return predictions



DEFAULT_EXCLUDE_FROM_NORM = [
    "Census_Both",
    "Census_Men",
    "Census_Women",
    "TotalCapacityOtherShelters",
    "CensusLTS",
    "Date",
    "Month",
    "Week",
]

def make_shelter_forecast(
    df_raw,
    horizon_days=14,
    target_col="Census_Men",
    covariance_threshold=0.8,
):
    """
    Thin wrapper around `main` that:

    - runs your full cleaning + feature-engineering + model pipeline
    - returns a tidy DataFrame with columns: `date`, `forecast`
    """
    preds = main(
        df=df_raw.copy(),
        target_col=target_col,
        day_ahead=horizon_days,
        covariance_threshhold=covariance_threshold,
        exclude_from_normalization=DEFAULT_EXCLUDE_FROM_NORM,
    )

    # Standardize column names for the app
    preds = preds.rename(columns={
        "Date": "date",
        "Predicted": "forecast",
    })

    return preds




