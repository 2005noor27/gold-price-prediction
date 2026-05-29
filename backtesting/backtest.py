"""
backtesting/backtest.py
=======================
Walk-forward honest backtesting engine.

Key principle: all metrics are computed on OUT-OF-SAMPLE predictions only
(Walk-Forward / TimeSeriesSplit). The model never sees test data during
training of any fold.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


def run_walk_forward(
    X: np.ndarray,
    y_ret: np.ndarray,
    prices: np.ndarray,
    dates: np.ndarray,
    make_model_fn,
    model_name: str,
    n_splits: int = 5,
) -> dict:
    """
    Run Walk-Forward Validation across n_splits folds.

    Parameters
    ----------
    X            : Feature matrix (n_samples, n_features).
    y_ret        : Target daily return % (n_samples,).
    prices       : Raw price series aligned with X (for reconstruction).
    dates        : Date array aligned with X.
    make_model_fn: Callable() → sklearn-compatible estimator.
    model_name   : Name string (used to detect MLP scaling need).
    n_splits     : Number of TimeSeriesSplit folds.

    Returns
    -------
    dict with keys:
      fold_metrics   : list of per-fold metric dicts
      all_dates      : concatenated test dates
      all_prices     : concatenated actual prices
      all_preds_ret  : concatenated predicted returns
      all_preds_price: concatenated reconstructed prices
      all_y_ret      : concatenated actual returns
      last_model     : the model fitted on the last fold
      avg_*          : scalar averages across folds
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_metrics   = []
    all_dates      = []
    all_prices     = []
    all_preds_ret  = []
    all_preds_price = []
    all_y_ret      = []
    last_model     = None

    needs_scaling  = "MLP" in model_name or "Neural" in model_name

    for fold_i, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y_ret[tr_idx], y_ret[te_idx]
        p_te       = prices[te_idx]
        d_te       = dates[te_idx]

        mdl = make_model_fn()

        if needs_scaling:
            sx, sy = StandardScaler(), StandardScaler()
            mdl.fit(sx.fit_transform(X_tr),
                    sy.fit_transform(y_tr.reshape(-1, 1)).ravel())
            preds = sy.inverse_transform(
                mdl.predict(sx.transform(X_te)).reshape(-1, 1)
            ).ravel()
        else:
            mdl.fit(X_tr, y_tr)
            preds = mdl.predict(X_te)

        # 1-step-ahead price reconstruction
        pp = np.zeros(len(p_te))
        pp[0] = p_te[0]
        for i in range(1, len(p_te)):
            pp[i] = p_te[i - 1] * (1 + preds[i] / 100)

        da   = float(np.mean(np.sign(y_te[1:]) == np.sign(preds[1:])) * 100)
        rmse = float(np.sqrt(mean_squared_error(p_te[1:], pp[1:])))
        mae  = float(mean_absolute_error(p_te[1:], pp[1:]))
        r2   = float(r2_score(y_te, preds))

        fold_metrics.append({
            "Fold": fold_i + 1,
            "Dir Acc %": round(da, 1),
            "RMSE $":    round(rmse, 2),
            "MAE $":     round(mae, 2),
            "R² (ret)":  round(r2, 4),
            "Test days": len(te_idx),
        })

        all_dates.append(d_te)
        all_prices.append(p_te)
        all_preds_ret.append(preds)
        all_preds_price.append(pp)
        all_y_ret.append(y_te)
        last_model = mdl

    all_dates_cat   = np.concatenate(all_dates)
    all_prices_cat  = np.concatenate(all_prices)
    all_predr_cat   = np.concatenate(all_preds_ret)
    all_predp_cat   = np.concatenate(all_preds_price)
    all_y_cat       = np.concatenate(all_y_ret)

    return dict(
        fold_metrics    = fold_metrics,
        all_dates       = all_dates,
        all_prices      = all_prices,
        all_preds_ret   = all_preds_ret,
        all_preds_price = all_preds_price,
        all_y_ret       = all_y_ret,
        last_model      = last_model,
        # concatenated
        all_dates_cat   = all_dates_cat,
        all_prices_cat  = all_prices_cat,
        all_predr_cat   = all_predr_cat,
        all_predp_cat   = all_predp_cat,
        all_y_cat       = all_y_cat,
        # averages
        avg_da   = float(np.mean([f["Dir Acc %"] for f in fold_metrics])),
        avg_rmse = float(np.mean([f["RMSE $"]    for f in fold_metrics])),
        avg_mae  = float(np.mean([f["MAE $"]     for f in fold_metrics])),
    )


def compute_strategy_metrics(
    pred_returns: np.ndarray,
    actual_returns: np.ndarray,
    threshold: float = 0.05,
    periods: int = 252,
) -> dict:
    """
    Compute full backtest metrics for a long-only strategy.

    Strategy: go long (hold gold) when predicted return > threshold,
    else stay in cash.

    Returns dict with: total_ret, ann_ret, max_dd, sharpe, sortino,
                       calmar, win_rate, n_trades, avg_win, avg_loss,
                       coverage, cum_strategy, cum_bh, dates_aligned.
    """
    position  = (pred_returns > threshold).astype(float)
    strat_r   = position * actual_returns
    bh_r      = actual_returns

    cum_s  = np.cumprod(1 + strat_r / 100)
    cum_bh = np.cumprod(1 + bh_r    / 100)

    def _metrics(rets, cum):
        n       = max(len(rets), 1)
        tot     = float((cum[-1] - 1) * 100)
        ann     = float((cum[-1] ** (periods / n) - 1) * 100)
        peak    = np.maximum.accumulate(cum)
        dd      = (cum - peak) / np.where(peak > 1e-9, peak, 1e-9) * 100
        max_dd  = float(dd.min())
        mu, sig = np.mean(rets), np.std(rets)
        sharpe  = float(mu / sig * np.sqrt(periods)) if sig > 1e-9 else 0.0
        down    = np.std(rets[rets < 0]) if np.any(rets < 0) else 1e-9
        sortino = float(mu / down * np.sqrt(periods)) if down > 1e-9 else 0.0
        calmar  = float(ann / abs(max_dd)) if abs(max_dd) > 1e-9 else 0.0
        return dict(tot=tot, ann=ann, dd=max_dd,
                    sharpe=sharpe, sortino=sortino, calmar=calmar)

    ms = _metrics(strat_r, cum_s)
    bh = _metrics(bh_r,    cum_bh)

    trade_mask = position > 0
    trade_rets = actual_returns[trade_mask]
    win_rate   = float(np.mean(trade_rets > 0) * 100) if len(trade_rets) > 0 else 0.0
    n_trades   = int(np.sum(np.diff(np.concatenate([[0], trade_mask.astype(int)])) == 1))
    avg_win    = float(np.mean(trade_rets[trade_rets > 0])) if np.any(trade_rets > 0) else 0.0
    avg_loss   = float(np.mean(trade_rets[trade_rets < 0])) if np.any(trade_rets < 0) else 0.0
    coverage   = int(np.mean(position) * 100)

    return dict(
        strategy=ms, bh=bh,
        win_rate=win_rate, n_trades=n_trades,
        avg_win=avg_win, avg_loss=avg_loss,
        coverage=coverage,
        cum_s=cum_s, cum_bh=cum_bh,
        strat_r=strat_r, bh_r=bh_r,
    )
