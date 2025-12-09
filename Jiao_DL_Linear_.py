#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd

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


# In[7]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

device = torch.device("cpu")
print("Using device:", device)

# ---------------- Data prep ----------------

def create_sequences(df, lookback, horizon, target_col):
    """df must contain Date + target_col; 'other census' already dropped."""
    df_no_date = df.drop(columns=["Date"])
    values = df_no_date.values
    target_idx = df_no_date.columns.get_loc(target_col)

    X, y = [], []
    for i in range(len(df_no_date) - lookback - horizon):
        X.append(values[i:i+lookback])
        y.append(values[i+lookback:i+lookback+horizon, target_idx])
    return np.array(X), np.array(y)


def prepare_datasets(
    df_clean,
    target_col,
    policy_cols,
    cyclic_cols,
    lookback=30,
    horizon=14,
    train_ratio=0.7,
    val_ratio=0.15
):
    # 1. Drop other Census_* columns so X does NOT see them
    other_census = [c for c in df_clean.columns
                    if c.startswith("Census_") and c != target_col]
    df_model = df_clean.drop(columns=other_census)

    # --- SAFETY FIX: Drop any rows that still have NaNs ---
    # This prevents the "Input contains NaN" crash if the cleaning pipeline missed something
    initial_len = len(df_model)
    df_model = df_model.dropna()
    dropped_count = initial_len - len(df_model)
    if dropped_count > 0:
        print(f"[prepare_datasets] ⚠️ Dropped {dropped_count} rows containing NaNs before modeling.")

    # 2. Identify scaling columns
    dont_scale = policy_cols + cyclic_cols + [target_col, "Date"]
    scale_cols = [c for c in df_model.columns if c not in dont_scale]

    # 3. Split Data
    n = len(df_model)
    if n < (lookback + horizon + 10): 
        # Safety check for very small datasets
        print("⚠️ Warning: Dataset is too small for the requested lookback/horizon.")
    
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    df_train = df_model.iloc[:train_end].reset_index(drop=True)
    df_val   = df_model.iloc[train_end:val_end].reset_index(drop=True)
    df_test  = df_model.iloc[val_end:].reset_index(drop=True)

    # 4. Fit Scaler (only on training data)
    scaler = StandardScaler()
    scaler.fit(df_train[scale_cols])

    def _apply(df):
        out = df.copy()
        out[scale_cols] = scaler.transform(df[scale_cols])
        return out

    df_train_s = _apply(df_train)
    df_val_s   = _apply(df_val)
    df_test_s  = _apply(df_test)

    # 5. Create Sequences
    X_train, y_train = create_sequences(df_train_s, lookback, horizon, target_col)
    X_val,   y_val   = create_sequences(df_val_s,   lookback, horizon, target_col)
    X_test,  y_test  = create_sequences(df_test_s,  lookback, horizon, target_col)

    meta = {
        "scaler": scaler,
        "scale_cols": scale_cols,
        "dont_scale": dont_scale,
        "target_col": target_col,
        "dropped_census": other_census,
        "lookback": lookback,
        "horizon": horizon,
        "splits": {"train": len(df_train), "val": len(df_val), "test": len(df_test)},
        "df_model": df_model,   
        "df_clean": df_clean    
    }
    return X_train, y_train, X_val, y_val, X_test, y_test, meta
# ---------------- DL Linear ----------------

class DLLinearForecaster(nn.Module):
    def __init__(self, input_dim, lookback, horizon):
        super().__init__()
        self.linear = nn.Linear(lookback * input_dim, horizon)

    def forward(self, x):
        B, L, F = x.shape
        return self.linear(x.reshape(B, L * F))


def train_dl_linear(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    learning_rate=1e-3,
    weight_decay=1e-4,
    batch_size=128,
    epochs=200,
    early_stopping_patience=20,
    run_name="DLLinear",
    # --- NEW ARGUMENTS ADDED HERE ---
    model_type="linear",
    hidden_dim=64
):
    input_dim = X_train.shape[2]
    lookback = X_train.shape[1]
    horizon = y_train.shape[1]

    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    X_val_t   = torch.from_numpy(X_val).float().to(device)
    y_val_t   = torch.from_numpy(y_val).float().to(device)
    X_test_t  = torch.from_numpy(X_test).float().to(device)
    y_test_t  = torch.from_numpy(y_test).float().to(device)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=batch_size, shuffle=False)

    # --- UPDATED MODEL SELECTION LOGIC ---

    model = DLLinearForecaster(input_dim, lookback, horizon).to(device)
        
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def eval_loader(loader):
        model.eval()
        ys, yhats = [], []
        with torch.no_grad():
            for Xb, yb in loader:
                yh = model(Xb)
                ys.append(yb.cpu().numpy())
                yhats.append(yh.cpu().numpy())
        y_true = np.concatenate(ys, axis=0)
        y_pred = np.concatenate(yhats, axis=0)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return mae, rmse, y_true, y_pred

    best_val_rmse = np.inf
    best_state = None
    patience = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            yh = model(Xb)
            loss = criterion(yh, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * Xb.size(0)

        train_mse = running_loss / len(train_loader.dataset)
        val_mae, val_rmse, _, _ = eval_loader(val_loader)

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            # Only print if we are running locally (optional)
            pass

        if val_rmse < best_val_rmse - 1e-4:
            best_val_rmse = val_rmse
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_mae, train_rmse, ytr_true, ytr_pred = eval_loader(train_loader)
    val_mae,   val_rmse,   yv_true,  yv_pred  = eval_loader(val_loader)
    test_mae,  test_rmse,  yte_true, yte_pred = eval_loader(test_loader)

    results = {
        "run_name": run_name,
        "train_MAE": train_mae,
        "train_RMSE": train_rmse,
        "val_MAE": val_mae,
        "val_RMSE": val_rmse,
        "test_MAE": test_mae,
        "test_RMSE": test_rmse,
    }
    preds = {
        "train_true": ytr_true, "train_pred": ytr_pred,
        "val_true":   yv_true,  "val_pred":  yv_pred,
        "test_true":  yte_true, "test_pred": yte_pred,
    }

    print(f"[{run_name}] Test MAE={test_mae:.3f}, RMSE={test_rmse:.3f}")
    return model, results, preds


# In[8]:


# ---------------- Main function----------------

def run_dl_linear_experiment(
    df_raw,
    target_col="Census_Men",
    diff_threshold=100,
    covariance_threshold=0.8,
    policy_cols=None,
    cyclic_cols=None,
    lookback=30,
    horizon=14,
    learning_rate=1e-3,
    weight_decay=1e-4,
    batch_size=128,
    epochs=200,
    early_stopping_patience=20,
    run_name=None
):
    # df_raw -> df_clean (you read CSV before calling this)
    df_clean = preprocess_shelter_df(
        df_raw,
        target_col=target_col,
        diff_threshold=diff_threshold,
        covariance_threshold=covariance_threshold
    )

    if policy_cols is None:
        policy_cols = [
            "CovidStayAtHomeMA",
            "CovidStateEmergencyMA",
            "CHNVParole",
            "Ordinance1373",
            "CARESHousing"
        ]
    if cyclic_cols is None:
        cyclic_cols = ["Month_sin", "Month_cos", "dow_sin", "dow_cos"]
    if run_name is None:
        run_name = f"DLLinear_{target_col}"

    X_train, y_train, X_val, y_val, X_test, y_test, meta = prepare_datasets(
        df_clean=df_clean,
        target_col=target_col,
        policy_cols=policy_cols,
        cyclic_cols=cyclic_cols,
        lookback=lookback,
        horizon=horizon
    )

    print(f"\nTarget: {target_col}")
    print("Dropped census cols from X:", meta["dropped_census"])
    print("Sequences:",
          "X_train", X_train.shape,
          "X_val",   X_val.shape,
          "X_test",  X_test.shape)

    model, results, preds = train_dl_linear(
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        run_name=run_name
    )

    return model, results, preds, meta




def make_dl_forecast(
    df_raw,
    horizon_days=14,
    target_col="Census_Men",
    lookback=30,
    learning_rate=1e-3,
    epochs=150,
    confidence_level=0.90,
    model_type="linear"
):
    """
    Wrapper function for Streamlit.
    """
    
    # 1. Clean and Preprocess
    df_clean = preprocess_shelter_df(
        df_raw,
        target_col=target_col,
        diff_threshold=100,
        covariance_threshold=0.8
    )

    # 2. Prepare Data
    policy_cols = [
        "CovidStayAtHomeMA", "CovidStateEmergencyMA",
        "CHNVParole", "Ordinance1373", "CARESHousing"
    ]
    cyclic_cols = ["Month_sin", "Month_cos", "dow_sin", "dow_cos"]

    X_train, y_train, X_val, y_val, X_test, y_test, meta = prepare_datasets(
        df_clean=df_clean,
        target_col=target_col,
        policy_cols=policy_cols,
        cyclic_cols=cyclic_cols,
        lookback=lookback,
        horizon=horizon_days,
        train_ratio=0.8,
        val_ratio=0.1
    )
    
    # 3. Train the Model
    # Select architecture based on user input
    try:
        if model_type == "mlp":
            model = MLPForecaster(X_train.shape[2], lookback, horizon_days, hidden_dim=64).to(device)
        else:
            model = DLLinearForecaster(X_train.shape[2], lookback, horizon_days).to(device)
    except NameError:
         # Fallback if classes aren't defined
         model = DLLinearForecaster(X_train.shape[2], lookback, horizon_days).to(device)

    model, results, preds = train_dl_linear(
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        learning_rate=learning_rate,
        epochs=epochs,
        early_stopping_patience=20,
        run_name="App_Forecast",
        # Pass these through if your training function accepts them, otherwise remove
        model_type=model_type, 
        hidden_dim=64
    )

    # 4. Generate Future Forecast Point Prediction
    df_model = meta["df_model"] 
    scaler = meta["scaler"]
    scale_cols = meta["scale_cols"]
    
    # Re-scale the full df_model to get the very last window
    df_model_scaled = df_model.copy()
    df_model_scaled[scale_cols] = scaler.transform(df_model[scale_cols])
    
    # Extract the last 'lookback' days 
    last_window = df_model_scaled.drop(columns=["Date"]).iloc[-lookback:].values
    input_tensor = torch.from_numpy(last_window).float().unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        forecast_raw = model(input_tensor).cpu().numpy().flatten() 

    # 5. Calculate DYNAMIC Error Bars (Per-Step Conformal Prediction)
    # Residuals shape: (N_Test_Samples, Horizon)
    test_residuals = preds["test_true"] - preds["test_pred"]
    
    # Calculate quantiles along axis=0 (down the column)
    # This gives us a specific error band for Day 1, Day 2, ... Day 14
    alpha = 1.0 - confidence_level
    q_lower = np.quantile(test_residuals, alpha / 2.0, axis=0)       # Shape: (Horizon,)
    q_upper = np.quantile(test_residuals, 1.0 - alpha / 2.0, axis=0) # Shape: (Horizon,)

    # 6. Construct Results DataFrame
    last_date = pd.to_datetime(df_clean["Date"].max())
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, horizon_days + 1)]

    # We add the per-step quantile array to the per-step forecast array
    # Python handles the element-wise addition automatically
    forecast_df = pd.DataFrame({
        "date": future_dates,
        "Shelter Guests": np.round(forecast_raw, 0),
        "Predicted_lower": np.round(forecast_raw + q_lower, 0),
        "Predicted_upper": np.round(forecast_raw + q_upper, 0)
    })
    
    # Clip negative values
    forecast_df["Predicted_lower"] = forecast_df["Predicted_lower"].clip(lower=0)
    forecast_df["Shelter Guests"] = forecast_df["Shelter Guests"].clip(lower=0)
    
    # Add metrics to return dict
    results["confidence_level"] = confidence_level

    return forecast_df, results




if __name__ == "__main__":
    # This block is for testing purposes and is not run by the app in any important way
    # =========================================================
    # 1. SETUP PLOTTING (Headless / No-GUI support)
    # =========================================================
    import matplotlib
    # Force matplotlib to use a 'non-interactive' backend 
    # This prevents the "_tkinter.TclError" you saw earlier
    matplotlib.use('Agg') 
    import matplotlib.pyplot as plt

    # =========================================================
    # 2. DEFINE EVALUATION FUNCTION
    # =========================================================
    def evaluate_and_plot_test_set(
        df_raw, 
        target_col="Census_Men", 
        train_split=0.70
    ):
        print(f"\n📊 RUNNING BACKTEST (Train on first {train_split*100:.0f}% of data)...")
        
        # 1. Clean Data
        df_clean = preprocess_shelter_df(df_raw, target_col=target_col)

        # 2. Prepare Datasets
        test_split = 1.0 - train_split
        val_split = test_split / 2
        
        X_train, y_train, X_val, y_val, X_test, y_test, meta = prepare_datasets(
            df_clean=df_clean,
            target_col=target_col,
            policy_cols=["CovidStayAtHomeMA", "CovidStateEmergencyMA", "CHNVParole", "Ordinance1373", "CARESHousing"],
            cyclic_cols=["Month_sin", "Month_cos", "dow_sin", "dow_cos"],
            lookback=30,
            horizon=14,
            train_ratio=train_split,
            val_ratio=val_split
        )
        

        model = DLLinearForecaster(X_train.shape[2], 30, 14).to(device)
        print("✅ Using Linear Architecture")

        # 4. Train
        model, results, preds = train_dl_linear(
            X_train, y_train, X_val, y_val, X_test, y_test,
            epochs=100, 
            batch_size=32,
            run_name="Backtest"
        )

        # 5. Generate Predictions
        model.eval()
        with torch.no_grad():
            X_test_t = torch.from_numpy(X_test).float().to(device)
            y_pred_raw = model(X_test_t).cpu().numpy()

        # 6. Extract Data (No unscaling needed as target was excluded from scaler)
        actuals = y_test[:, 0]
        predictions = y_pred_raw[:, 0]

        # 7. Plotting
        plt.figure(figsize=(14, 7))
        plt.plot(actuals, label='Actual Census', color='black', alpha=0.7, linewidth=2)
        plt.plot(predictions, label='Model Prediction (1-day ahead)', color='#007bff', linewidth=2)
        
        plt.title(f"Backtest Results: {target_col}\n(Model trained on first {train_split*100:.0f}% of history)", fontsize=16)
        plt.ylabel("Guests", fontsize=12)
        plt.xlabel("Days into Test Period", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # SAVE instead of SHOW (Fixes TclError)
        filename = 'backtest_results.png'
        plt.savefig(filename)
        print(f"\n✅ Plot saved successfully to: {filename}")
        plt.close()

    # =========================================================
    # 3. EXECUTE BACKTEST
    # =========================================================
    # Ensure this path matches your file location
    df_raw = pd.read_csv('../BPHC Census Data 20251113.csv')
    evaluate_and_plot_test_set(df_raw, target_col="Census_Men", train_split=0.70)


    # =========================================================
    # 4. EXECUTE LEAKAGE CHECK
    # =========================================================
    target = "Census_Men"
    print(f"\n--- RUNNING LEAKAGE CHECK FOR {target} ---")

    # Run preprocessing
    df_clean = preprocess_shelter_df(
        df_raw,
        target_col=target,
        diff_threshold=100,
        covariance_threshold=0.8
    )

    # Run preparation
    _, _, _, _, _, _, meta = prepare_datasets(
        df_clean=df_clean,
        target_col=target,
        policy_cols=["CovidStayAtHomeMA", "CovidStateEmergencyMA", "CHNVParole", "Ordinance1373", "CARESHousing"],
        cyclic_cols=["Month_sin", "Month_cos", "dow_sin", "dow_cos"],
        lookback=30,
        horizon=14
    )

    df_model = meta["df_model"]
    used_columns = list(df_model.columns)
    
    print(f"\n✅ FINAL FEATURES USED BY MODEL ({len(used_columns)} total):")
    print(used_columns)

    # Check correlations
    print(f"\n🕵️‍♂️ DETECTIVE MODE: Checking for disguised leaks...")
    correlations = df_model.corr()[target].sort_values(ascending=False)
    correlations = correlations.drop(target, errors='ignore')
    
    suspicious = correlations[correlations > 0.90]
    
    if len(suspicious) > 0:
        print("\n⚠️ WARNING: The following features are HIGHLY correlated (> 0.90) with the target.")
        print(suspicious)
    else:
        print("\n✅ CLEAN: No suspiciously high correlations found (closest feature < 0.90).")

    # Check forbidden keywords
    forbidden = ["Both", "Capacity", "Women" if target=="Census_Men" else "Men"]
    print("\n🔍 CHECKING FOR FORBIDDEN KEYWORDS:")
    found_forbidden = [c for c in used_columns if any(bad in c for bad in forbidden)]
    
    if found_forbidden:
        print(f"❌ DANGER: Found potential leak columns: {found_forbidden}")
    else:
        print("✅ CLEAN: No obvious forbidden keywords found.")