

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor


# =========================================================
# 1. Spike cleaning with neighbor mean
# =========================================================
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
      3) abs(previous - next) <= diff_threshold (neighbors agree)

    When an outlier is found, its value is replaced by the mean of the
    nearest neighbor_window valid values before and after that index.

    NOTE: This uses future values in smoothing, which is fine because we
    treat this as data cleaning, not a forecasting operation.
    """

    if not inplace:
        df = df.copy()

    # Ensure sorted by date for temporal neighbors
    if date_col in df.columns:
        if not df[date_col].is_monotonic_increasing:
            print(f"[clean_with_neighbor_mean] {date_col} is not sorted. Sorting by {date_col}.")
            df = df.sort_values(date_col).reset_index(drop=True)

    n_rows = len(df)
    changes = []

    # Step 0: Replace zeros using neighbor means BEFORE spike cleaning
    for feature in features:
        if feature not in df.columns:
            print(f"[clean_with_neighbor_mean] Feature '{feature}' not found. Skipping zero replacement.")
            continue

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

            changes.append({
                "index": idx,
                "feature": feature,
                "old_value": 0,
                "new_value": new_val
            })
            df.at[idx, feature] = new_val

    # Step 1: Spike cleaning
    for feature in features:
        if feature not in df.columns:
            print(f"[clean_with_neighbor_mean] Feature '{feature}' not found. Skipping spike cleaning.")
            continue

        series = df[feature]

        for pos in range(1, n_rows - 1):
            current_val = series.iloc[pos]
            prev_val = series.iloc[pos - 1]
            next_val = series.iloc[pos + 1]

            if pd.isna(current_val) or pd.isna(prev_val) or pd.isna(next_val):
                continue

            big_jump_prev = abs(current_val - prev_val) > diff_threshold
            big_jump_next = abs(current_val - next_val) > diff_threshold

            if not (big_jump_prev and big_jump_next):
                continue

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
                continue

            new_val = float(np.round(np.mean(neighbor_vals), 0))

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


# =========================================================
# 2. Date → sin/cos seasonality
# =========================================================
def transform_dates_in_sin(df, date_col="Date"):
    """
    Create Month_sin, Month_cos, dow_sin, dow_cos from Date.
    Drops 'Month' and 'Week' if they exist, since they become redundant.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    month_number = df[date_col].dt.month
    dow = df[date_col].dt.weekday + 1  # Monday=1 ... Sunday=7

    month_sin = np.sin(2 * np.pi * month_number / 12).round(4)
    month_cos = np.cos(2 * np.pi * month_number / 12).round(4)
    dow_sin = np.sin(2 * np.pi * dow / 7).round(4)
    dow_cos = np.cos(2 * np.pi * dow / 7).round(4)

    df["Month_sin"] = month_sin
    df["Month_cos"] = month_cos
    df["dow_sin"] = dow_sin
    df["dow_cos"] = dow_cos

    # Drop raw Month/Week columns if present
    for col in ["Month", "Week"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


# =========================================================
# 3. Drop explicit columns by name
# =========================================================
def dropping_features(df, features):
    df = df.copy()
    drop_existing = [f for f in features if f in df.columns]
    if len(drop_existing) > 0:
        df = df.drop(columns=drop_existing)
        print(f"[dropping_features] Dropped: {drop_existing}")
    else:
        print("[dropping_features] No matching features to drop.")
    return df


# =========================================================
# 4. Drop highly correlated features (with protection)
# =========================================================
def drop_high_covariance_features(df, threshold=0.8, targets=None, protect_cols=None):
    """
    Drop highly correlated features.

    targets: list of target column names (e.g., ["Census_Men", "Census_Women"])
    protect_cols: features that should NEVER be dropped (e.g., policy dummies + targets)
    """
    if targets is None:
        targets = ["Census_Men"]
    if protect_cols is None:
        protect_cols = []

    data = df.copy()
    date_series = pd.to_datetime(data["Date"])
    df_no_date = data.drop(columns="Date", axis=1)

    # Exclude ALL targets from feature list
    features = [c for c in df_no_date.columns if c not in targets]

    corr = df_no_date[features].corr()

    # Calculate correlation with ALL targets, take max abs corr
    target_corr_dict = {}
    for feat in features:
        correlations = [abs(df_no_date[feat].corr(df_no_date[t])) for t in targets if t in df_no_date.columns]
        target_corr_dict[feat] = max(correlations) if len(correlations) > 0 else 0.0

    to_drop = set()

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            a, b = features[i], features[j]

            if a in to_drop or b in to_drop:
                continue
            if a in protect_cols or b in protect_cols:
                continue

            if abs(corr.loc[a, b]) > threshold:
                # Keep the feature more correlated with any target
                if target_corr_dict[a] >= target_corr_dict[b]:
                    to_drop.add(b)
                else:
                    to_drop.add(a)

    print(f"[drop_high_covariance_features] Dropping {len(to_drop)} features: {to_drop}")

    df_reduced = df_no_date.drop(columns=list(to_drop), errors="ignore")
    df_reduced.insert(0, "Date", date_series)

    return df_reduced


# =========================================================
# 5. Fill NA for selected numeric time-series columns
# =========================================================
def fill_NA_features(df, cols=None):
    """
    Forward-fill and backward-fill selected time-series numeric columns.
    Does NOT modify any other columns.
    """
    if cols is None:
        cols = ['UnemploymentRate', 'ZHVIMidBOS', 'TempMaxF', 'PrecipInch', 'SnowInch', 'ZHVIMidUS']

    df = df.copy()

    for col in cols:
        if col in df.columns:
            before = df[col].isna().sum()
            df[col] = df[col].ffill().bfill()
            after = df[col].isna().sum()
            print(f"[fill_NA_features] {col}: filled {before - after} values (remaining: {after})")
        else:
            print(f"[fill_NA_features] Column {col} not found in dataframe.")

    return df


# =========================================================
# 6. Main cleaning pipeline
# =========================================================
def preprocess_shelter_df(
    df,
    target_col="Census_Men",
    diff_threshold=100,
    covariance_threshold=0.8
):
    """
    Full cleaning pipeline:
      1) Ensure Date is datetime & sorted
      2) Clean spikes in key occupancy/capacity features
      3) Create sin/cos date features
      4) Drop some explicitly unwanted features
      5) Drop highly correlated features (protecting census targets + policies)
      6) Fill NAs in key numeric variables

    Returns a cleaned dataframe, with targets kept in original scale.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # 1) Spike cleaning for core occupancy/capacity features
    features_to_clean = [
        "Census_Men",
        "Census_Women",
        "Census_Both",
        "TotalCapacity_Men",
        "TotalCapacity_Women",
        "TotalCapacity_Both",
        "TotalCapacityOtherShelters"
    ]
    df_clean, log_changes = clean_with_neighbor_mean(
        df,
        features=features_to_clean,
        diff_threshold=diff_threshold,
        neighbor_window=2,
        date_col="Date",
        inplace=False
    )
    print(f"[preprocess_shelter_df] Spike cleaning applied, {len(log_changes)} changes logged.")

    # 2) Date → sin/cos and drop raw Month/Week
    df_seasonal = transform_dates_in_sin(df_clean, date_col="Date")

    # 3) Drop explicitly unwanted features
    drop_features = [
        "TotalCapacity_Men", "TotalCapacity_Women", "Census_Both",
        "TotalCapacity_Both", "MFRForSaleBOS", "NewUnitsApproved",
        "NoticeToQuitTotal", "EvictionFilings", "EvictionExecutions", "SFRForSaleBOS",'HomesForSaleBOS'
    ]
    df_dropped = dropping_features(df_seasonal, drop_features)

    # 4) Drop highly correlated features, protecting census targets + policy dummies
    policy_cols = [
        "CovidStayAtHomeMA", "CovidStateEmergencyMA",
        "CHNVParole", "Ordinance1373", "CARESHousing"
    ]

    # list of all census targets we want to keep structurally
    census_targets_all = ["Census_Men", "Census_Women"]
    targets = [c for c in census_targets_all if c in df_dropped.columns]

    protect_cols = policy_cols + targets

    df_no_cov = drop_high_covariance_features(
        df_dropped,
        threshold=covariance_threshold,
        targets=targets,
        protect_cols=protect_cols
    )

    # 5) Fill NAs in key numeric features
    df_no_na = fill_NA_features(df_no_cov)

    return df_no_na


def compute_rmse_and_c(y_true, y_pred, n_last=1000):
    """
    Compute RMSE and c_66, c_95 on the last n_last points.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    if n_last is not None and len(y_true) > n_last:
        y_true = y_true[-n_last:]
        y_pred = y_pred[-n_last:]

    errors = np.abs(y_true - y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    c_66 = np.quantile(errors, 0.66)
    c_95 = np.quantile(errors, 0.95)

    return rmse, c_66, c_95

def backtest_ensemble_and_forecast(
    df_clean,
    target_col="Census_Men",
    days_ahead=14,
    n_test=1000
):
    """
    Backtest RF+XGB ensemble on last n_test targets and produce a 14 day forecast.
    Returns: forecast_df, rmse, c66, c95
    """

    df_model = df_clean.copy().sort_values("Date").reset_index(drop=True)
    df_model["Date"] = pd.to_datetime(df_model["Date"])

    future_col = f"{target_col}_future"
    df_model[future_col] = df_model[target_col].shift(-days_ahead)
    df_model = df_model.dropna(subset=[future_col]).reset_index(drop=True)

    # split train / test
    if len(df_model) <= n_test:
        n_test = max(1, len(df_model) // 3)

    train_df = df_model.iloc[:-n_test].copy()
    test_df  = df_model.iloc[-n_test:].copy()

    X_train = train_df.drop(columns=["Date", future_col])
    X_train = X_train.select_dtypes(include=[np.number])
    y_train = train_df[future_col].values

    X_test = test_df.drop(columns=["Date", future_col])
    X_test = X_test.select_dtypes(include=[np.number])
    y_test = test_df[future_col].values

    # train XGB
    xgb = XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7]
    }
    grid_search = GridSearchCV(
        xgb,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    best_xgb = grid_search.best_estimator_

    # train RF
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    param_dist = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_leaf": [5, 10],
    }
    tscv = TimeSeriesSplit(n_splits=5)
    rs = RandomizedSearchCV(
        rf,
        param_dist,
        n_iter=20,
        cv=tscv,
        scoring="neg_mean_squared_error",
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    rs.fit(X_train, y_train)
    best_rf = rs.best_estimator_

    # backtest predictions
    y_pred_test_xgb = best_xgb.predict(X_test)
    y_pred_test_rf  = best_rf.predict(X_test)
    y_pred_test     = 0.35 * y_pred_test_xgb + 0.65 * y_pred_test_rf

    rmse_ens, c66_ens, c95_ens = compute_rmse_and_c(y_test, y_pred_test, n_last=n_test)

    # forecast next 14 days
    X_future = df_model.drop(columns=["Date", future_col]).iloc[-days_ahead:]
    X_future_num = X_future.select_dtypes(include=[np.number])

    y_pred_future_xgb = best_xgb.predict(X_future_num)
    y_pred_future_rf  = best_rf.predict(X_future_num)
    y_pred_future     = 0.35 * y_pred_future_xgb + 0.65 * y_pred_future_rf

    last_dates = pd.to_datetime(df_model["Date"].iloc[-days_ahead:]) + pd.to_timedelta(days_ahead, unit="D")

    forecast_df = pd.DataFrame(
        {
            "date": last_dates,
            "Shelter Guests": np.round(y_pred_future, 0)
        }
    ).reset_index(drop=True)

    return forecast_df, rmse_ens, c66_ens, c95_ens


def ar_trend_backtest_and_forecast(
    df_clean,
    target_col="Census_Men",
    days_ahead=14,
    test_size=15,
    train_window=1000,
    alpha=0.1,
    n_last=1000
):
    """
    AR Lasso with lags/diffs:
      1) Walkforward backtest to compute RMSE, c66, c95 on last n_last points
      2) Fit final model on full history and forecast next days_ahead values

    Returns:
      forecast_df (next days_ahead),
      rmse_ar, c66_ar, c95_ar
    """

    # ---------- 1. Walkforward backtest ----------
    work = df_clean.copy().sort_values("Date").reset_index(drop=True)
    work["Date"] = pd.to_datetime(work["Date"])
    y = work[target_col].astype(float)

    future_col = f"{target_col}_future"
    work[future_col] = y.shift(-days_ahead)

    lags = (1, 2, 3, 7, 10, 14, 18, 21)
    for k in lags:
        work[f"lag_{k}"] = y.shift(k)

    work["diff_1"] = y - y.shift(1)
    work["diff_7"] = y - y.shift(7)

    feature_cols = [f"lag_{k}" for k in lags] + ["diff_1", "diff_7"]

    work = work.dropna(subset=[future_col] + feature_cols).reset_index(drop=True)

    X = work[feature_cols].values
    y_future = work[future_col].values
    y_naive  = work[target_col].values
    dates_all = work["Date"].dt.tz_localize(None).to_numpy() + np.timedelta64(days_ahead, "D")

    n = len(work)
    gap = days_ahead

    if n < train_window + gap + test_size:
        raise ValueError(f"Not enough data: n={n}, need >= {train_window + gap + test_size}")

    all_y_true = []
    all_y_pred = []
    all_y_naive = []
    all_dates = []

    start = 0
    while True:
        train_end  = start + train_window
        test_start = train_end + gap
        test_end   = test_start + test_size

        if test_end > n:
            break

        train_start = start
        X_train = X[train_start:train_end]
        y_train = y_future[train_start:train_end]

        X_test  = X[test_start:test_end]
        y_test  = y_future[test_start:test_end]
        y_naive_test = y_naive[test_start:test_end]
        dates_test   = dates_all[test_start:test_end]

        model = Lasso(alpha=alpha, fit_intercept=True, max_iter=20000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        all_y_true.append(y_test)
        all_y_pred.append(y_pred)
        all_y_naive.append(y_naive_test)
        all_dates.append(dates_test)

        start += test_size

    all_y_true  = np.concatenate(all_y_true)
    all_y_pred  = np.concatenate(all_y_pred)
    all_y_naive = np.concatenate(all_y_naive)
    all_dates   = np.concatenate(all_dates)

    # do NOT round before computing metrics to keep it fair
    predictions_df = pd.DataFrame(
        {
            "Date": all_dates,
            "Actual": all_y_true,
            "Naive": all_y_naive,
            "AR_trend_pred": all_y_pred,
        }
    ).sort_values("Date").reset_index(drop=True)

    rmse_ar, c66_ar, c95_ar = compute_rmse_and_c(
        predictions_df["Actual"].values,
        predictions_df["AR_trend_pred"].values,
        n_last=n_last
    )

    # ---------- 2. Final model for future forecast ----------
    df_full = df_clean.copy().sort_values("Date").reset_index(drop=True)
    df_full["Date"] = pd.to_datetime(df_full["Date"])
    y_full = df_full[target_col].astype(float).reset_index(drop=True)

    work_forecast = pd.DataFrame({"y": y_full})
    for k in lags:
        work_forecast[f"lag_{k}"] = y_full.shift(k)
    work_forecast["diff_1"] = y_full - y_full.shift(1)
    work_forecast["diff_7"] = y_full - y_full.shift(7)

    future_col = f"{target_col}_future"
    work_forecast[future_col] = y_full.shift(-days_ahead)

    work_forecast = work_forecast.dropna(subset=feature_cols + [future_col]).reset_index(drop=True)

    X_train_full = work_forecast[feature_cols].values
    y_train_full = work_forecast[future_col].values

    final_model = Lasso(alpha=alpha, random_state=42, max_iter=20000)
    final_model.fit(X_train_full, y_train_full)

    # features for last days_ahead times (NaN-safe)
    feature_df_full = pd.DataFrame(index=df_full.index)
    for k in lags:
        feature_df_full[f"lag_{k}"] = y_full.shift(k)
    feature_df_full["diff_1"] = y_full - y_full.shift(1)
    feature_df_full["diff_7"] = y_full - y_full.shift(7)

    # drop rows with any NaNs in the feature set
    feature_df_full = feature_df_full.dropna(subset=feature_cols)

    if feature_df_full.empty:
        raise ValueError("No valid rows available for AR trend forecast (all X_future rows contain NaNs).")

    # take the last `days_ahead` valid rows
    X_future = feature_df_full[feature_cols].iloc[-days_ahead:]
    future_dates = df_full.loc[X_future.index, "Date"] + pd.to_timedelta(days_ahead, unit="D")

    y_pred_future = final_model.predict(X_future.values)

    forecast_df = pd.DataFrame(
        {
            "date": future_dates.dt.tz_localize(None),
            "Shelter Guests": np.round(y_pred_future, 0)
        }
    ).reset_index(drop=True)

    return forecast_df, rmse_ar, c66_ar, c95_ar


def model_score(rmse, c66):
    return 1.0 * rmse + 1.0 * c66


def make_classical_forecast(
    df_raw,
    target_col="Census_Men",
    days_ahead=14,
    n_test=1000
):
    """
    Cleans raw data, runs ensemble and ARIMA models,
    compares them on last n_test days, and returns:
      forecast_df (14 rows with date, Shelter Guests, lower, upper, model name)
      metrics dict with rmse and c for best model.
    """

    # clean once
    df_clean = preprocess_shelter_df(
        df_raw,
        target_col=target_col,
        diff_threshold=100,
        covariance_threshold=0.8
    )

    # ensemble
    ens_forecast, rmse_ens, c66_ens, c95_ens = backtest_ensemble_and_forecast(
        df_clean,
        target_col=target_col,
        days_ahead=days_ahead,
        n_test=n_test
    )
    score_ens = model_score(rmse_ens, c66_ens)

    # AR trend backtest and forecast (single function)
    ar_forecast, rmse_arima, c66_arima, c95_arima = ar_trend_backtest_and_forecast(
        df_clean,
        target_col=target_col,
        days_ahead=days_ahead,
        test_size=15,
        train_window=1000,
        alpha=0.1,
        n_last=n_test
    )
    score_arima = model_score(rmse_arima, c66_arima)


    if score_ens <= score_arima:
        best_name = "ensemble_RF_XGB"
        best_forecast = ens_forecast.copy()
        best_rmse = rmse_ens
        best_c66 = c66_ens
        best_c95 = c95_ens
    else:
        best_name = "arima_lasso"
        best_forecast = ar_forecast.copy()
        best_rmse = rmse_arima
        best_c66 = c66_arima
        best_c95 = c95_arima

    # add simple error band using c95
    best_forecast["Predicted_lower"] = np.round(np.maximum(best_forecast["Shelter Guests"] - best_c66, 0), 0)
    best_forecast["Predicted_upper"] = np.round(best_forecast["Shelter Guests"] + best_c66, 0)
    best_forecast["Model"] = best_name

    metrics = {
        "model": best_name,
        "RMSE_last_1000": best_rmse,
        "c66_last_1000": best_c66,
        "c95_last_1000": best_c95,
        "score": model_score(best_rmse, best_c66),
    }

    return best_forecast, metrics
