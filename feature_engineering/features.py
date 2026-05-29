"""
feature_engineering/features.py
================================
Builds time-series features for ML models:
  1. Lag features (return lags, stationary)
  2. Rolling statistics (mean, std)
  3. Volatility features (realized vol, vol ratio)
  4. Momentum features (ROC, z-score, price momentum)
"""

import numpy as np
import pandas as pd

# ── Fixed lag days that are always added ──────────────────────────────────────
FIXED_LAGS = [7, 14, 21]

# ── Rolling windows ───────────────────────────────────────────────────────────
ROLL_WINDOWS = [5, 10, 14, 20, 30]

# ── Momentum horizons ─────────────────────────────────────────────────────────
MOM_HORIZONS = [5, 7, 10, 14, 21, 30]


def build_features(
    df: pd.DataFrame,
    n_lags: int = 5,
    price_col: str = "Price_Gold",
    return_col: str = "Return_Gold",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Add all time-series features to df (in-place copy) and return
    (enriched_df, feature_column_names).

    Parameters
    ----------
    df          : DataFrame with at least `price_col` and `return_col` columns.
    n_lags      : Number of slider-controlled lag features (1..n_lags).
    price_col   : Column name for price series.
    return_col  : Column name for daily return % series.

    Returns
    -------
    df          : DataFrame with new feature columns added.
    feature_cols: List of feature column names (in order).
    """
    df = df.copy()
    p  = df[price_col]
    r  = df[return_col]

    # ── 1. Lag features ───────────────────────────────────────────────────────
    lag_cols = []
    for lag in range(1, n_lags + 1):
        col = f"lag_{lag}"
        df[col] = r.shift(lag)
        lag_cols.append(col)

    for lag in FIXED_LAGS:
        if lag > n_lags:
            col = f"lag_{lag}"
            df[col] = r.shift(lag)
            lag_cols.append(col)

    # ── 2. Rolling statistics ─────────────────────────────────────────────────
    roll_cols = []
    for w in ROLL_WINDOWS:
        m_col = f"rolling_mean_{w}"
        s_col = f"rolling_std_{w}"
        df[m_col] = r.rolling(w).mean()
        df[s_col] = r.rolling(w).std()
        roll_cols.extend([m_col, s_col])

    # ── 3. Volatility features ────────────────────────────────────────────────
    vol_cols = []
    for w, label in [(5, "vol_5"), (21, "vol_21"), (63, "vol_63")]:
        df[label] = r.rolling(w).std() * np.sqrt(252)
        vol_cols.append(label)
    df["vol_ratio"] = df["vol_5"] / (df["vol_63"] + 1e-9)
    vol_cols.append("vol_ratio")

    # ── 4. Momentum features ──────────────────────────────────────────────────
    mom_cols = []
    for d in MOM_HORIZONS:
        col = f"mom_{d}d"
        df[col] = p.pct_change(d) * 100
        mom_cols.append(col)

    df["roc_5"]     = p.pct_change(5)  * 100
    df["roc_20"]    = p.pct_change(20) * 100
    sma20           = p.rolling(20).mean()
    std20           = p.rolling(20).std()
    df["zscore_20"] = (p - sma20) / (std20 + 1e-9)
    mom_cols.extend(["roc_5", "roc_20", "zscore_20"])

    feature_cols = lag_cols + roll_cols + vol_cols + mom_cols
    return df, feature_cols


def add_asset_returns(
    df: pd.DataFrame,
    price_feats: list[str],
    indic_feats: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Convert price-based extra features (Oil, Dollar, Stocks) to returns,
    and normalise VIX / TNX_Yield as % changes.

    Returns (df, extra_feature_cols).
    """
    df = df.copy()
    ret_cols = []

    for feat in price_feats:
        col = f"{feat}_ret"
        df[col] = df[feat].pct_change() * 100
        ret_cols.append(col)

    indic_out = []
    for feat in indic_feats:
        if feat in ("VIX", "TNX_Yield") and feat in df.columns:
            col = f"{feat}_ret"
            df[col] = df[feat].pct_change() * 100
            indic_out.append(col)
        elif feat in df.columns:
            indic_out.append(feat)

    return df, ret_cols + indic_out
