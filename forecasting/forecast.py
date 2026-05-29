"""
forecasting/forecast.py
=======================
Recursive multi-step forecasting with probabilistic output.

Key design choices:
  - Returns-based prediction (stationary target)
  - Recursive: each predicted return feeds the next step's lag features
  - Probabilistic: 500 Monte Carlo paths sampled from historical residuals
  - Outputs P10 / P25 / P50 / P75 / P90 percentiles
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def recursive_forecast(
    mdl,
    feature_cols: list[str],
    fc_df: pd.DataFrame,
    fc_lags: int,
    fc_days: int,
    ts_feat_cols: list[str],
    extra_ret_cols: list[str],
    extra_feats: list[str],
) -> tuple[list, list, float, list]:
    """
    Run the recursive forecasting loop.

    Returns
    -------
    forecast_dates  : list of pd.Timestamp
    forecast_prices : list of float (point estimates)
    cv_rmse         : float (cross-validated uncertainty estimate)
    cv_errors       : list of float (per-step CV errors in USD)
    """
    X_all      = fc_df[feature_cols].values
    y_all      = fc_df["Return_Gold"].values
    prices_arr = fc_df["Price_Gold"].values

    # TimeSeriesSplit CV for uncertainty estimation
    tscv      = TimeSeriesSplit(n_splits=5)
    cv_errors = []
    for tr_idx, te_idx in tscv.split(X_all):
        mdl.fit(X_all[tr_idx], y_all[tr_idx])
        preds_cv = mdl.predict(X_all[te_idx])
        for local_i, global_i in enumerate(te_idx):
            if global_i > 0:
                p_pred = prices_arr[global_i - 1] * (1 + preds_cv[local_i] / 100)
                cv_errors.append(abs(prices_arr[global_i] - p_pred))
    cv_rmse = float(np.mean(cv_errors)) if cv_errors else 50.0

    # Final fit on full data
    mdl.fit(X_all, y_all)

    ret_hist   = list(fc_df["Return_Gold"].values)
    price_hist = list(fc_df["Price_Gold"].values)
    last_extra = {
        c: fc_df[c].iloc[-1]
        for c in extra_ret_cols
        if c in fc_df.columns
    }
    last_date  = fc_df["Date"].iloc[-1]
    ts_vals    = [fc_df[c].iloc[-1] for c in ts_feat_cols if c in fc_df.columns]

    forecast_dates, forecast_prices = [], []
    for step in range(fc_days):
        lags_row  = [ret_hist[-(i)] for i in range(1, fc_lags + 1)]
        extra_row = [last_extra.get(c, 0) for c in extra_ret_cols]
        row       = np.array(lags_row + extra_row + ts_vals).reshape(1, -1)

        # Ensure row length matches feature_cols
        row = row[:, : len(feature_cols)]

        pred_ret   = float(np.clip(mdl.predict(row)[0], -5.0, 5.0))
        pred_price = price_hist[-1] * (1 + pred_ret / 100)

        next_date = last_date + pd.Timedelta(days=step + 1)
        while next_date.weekday() >= 5:
            next_date += pd.Timedelta(days=1)

        forecast_dates.append(next_date)
        forecast_prices.append(pred_price)
        ret_hist.append(pred_ret)
        price_hist.append(pred_price)
        last_date = next_date

    return forecast_dates, forecast_prices, cv_rmse, cv_errors


def monte_carlo_bands(
    forecast_prices: np.ndarray,
    fc_df: pd.DataFrame,
    cv_errors: list[float],
    fc_days: int,
    n_sim: int = 500,
    seed: int = 42,
) -> dict:
    """
    Run N Monte Carlo simulations to produce probabilistic forecast bands.

    Returns dict with p10, p25, p50, p75, p90 arrays and sim_paths matrix.
    """
    resid_src = fc_df["Return_Gold"].diff().dropna().values
    rng       = np.random.default_rng(seed)
    sim_paths = np.zeros((n_sim, fc_days))

    last_price = float(fc_df["Price_Gold"].iloc[-1])

    for s in range(n_sim):
        ph = [last_price]
        for step in range(fc_days):
            noise     = rng.choice(resid_src) if len(resid_src) > 0 else rng.normal(0, 0.5)
            target_r  = (forecast_prices[step] / ph[-1] * 100 - 100)
            sim_ret   = float(np.clip(
                rng.normal(target_r, abs(noise) * 0.5), -8, 8
            ))
            sim_p     = ph[-1] * (1 + sim_ret / 100)
            sim_paths[s, step] = sim_p
            ph.append(sim_p)

    return dict(
        p10 = np.percentile(sim_paths, 10, axis=0),
        p25 = np.percentile(sim_paths, 25, axis=0),
        p50 = np.percentile(sim_paths, 50, axis=0),
        p75 = np.percentile(sim_paths, 75, axis=0),
        p90 = np.percentile(sim_paths, 90, axis=0),
        sim_paths = sim_paths,
    )


def probability_of_gain(
    sim_paths: np.ndarray,
    last_actual: float,
) -> float:
    """Probability (%) that the final day forecast > last actual price."""
    return float(np.mean(sim_paths[:, -1] > last_actual) * 100)
