"""
tests/test_features.py
======================
Unit tests for feature engineering, backtesting, and utility modules.
Run with: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest


def _make_price_df(n: int = 300) -> pd.DataFrame:
    rng    = np.random.default_rng(42)
    prices = 2000 + np.cumsum(rng.normal(0, 10, n))
    dates  = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({"Date": dates, "Price_Gold": prices})
    df["Return_Gold"] = df["Price_Gold"].pct_change() * 100
    return df


# ── feature_engineering ───────────────────────────────────────────────────────

def test_build_features_returns_correct_types():
    from feature_engineering.features import build_features
    df = _make_price_df()
    enriched, cols = build_features(df, n_lags=5)
    assert isinstance(enriched, pd.DataFrame)
    assert isinstance(cols, list)
    assert len(cols) > 0


def test_build_features_column_count():
    from feature_engineering.features import build_features, FIXED_LAGS, ROLL_WINDOWS, MOM_HORIZONS
    df = _make_price_df()
    n_lags = 5
    _, cols = build_features(df, n_lags=n_lags)
    extra = len([l for l in FIXED_LAGS if l > n_lags])
    roll  = 2 * len(ROLL_WINDOWS)
    vol   = 4
    mom   = len(MOM_HORIZONS) + 3
    assert len(cols) >= n_lags + extra + roll + vol + mom


def test_build_features_no_all_nan():
    from feature_engineering.features import build_features
    df = _make_price_df(300)
    enriched, cols = build_features(df, n_lags=5)
    enriched_clean = enriched.dropna()
    for c in cols:
        if c in enriched_clean.columns:
            assert enriched_clean[c].notna().any(), f"{c} is all NaN"


def test_lag_1_is_previous_return():
    """lag_1[i] == Return_Gold[i-1] after dropna."""
    from feature_engineering.features import build_features
    df = _make_price_df(100)
    enriched, _ = build_features(df, n_lags=3)
    enriched = enriched.dropna().reset_index(drop=True)
    # lag_1 at row j should equal the Return_Gold value one original row earlier
    # Use diff-based verification: lag_1[j] == Return_Gold[j-1] in enriched
    for j in range(1, 5):
        assert abs(enriched["lag_1"].iloc[j] - enriched["Return_Gold"].iloc[j - 1]) < 1e-9


# ── backtesting ───────────────────────────────────────────────────────────────

def test_compute_strategy_metrics_keys():
    from backtesting.backtest import compute_strategy_metrics
    rng    = np.random.default_rng(0)
    preds  = rng.normal(0, 0.5, 200)
    actual = rng.normal(0, 0.5, 200)
    result = compute_strategy_metrics(preds, actual)
    for key in ("strategy", "bh", "win_rate", "n_trades", "cum_s", "cum_bh"):
        assert key in result


def test_win_rate_bounds():
    from backtesting.backtest import compute_strategy_metrics
    rng    = np.random.default_rng(1)
    preds  = rng.normal(0.2, 0.3, 300)
    actual = rng.normal(0.05, 0.5, 300)
    result = compute_strategy_metrics(preds, actual)
    assert 0.0 <= result["win_rate"] <= 100.0


def test_cum_equity_starts_near_one():
    from backtesting.backtest import compute_strategy_metrics
    rng    = np.random.default_rng(2)
    preds  = rng.normal(0, 0.3, 100)
    actual = rng.normal(0, 0.3, 100)
    result = compute_strategy_metrics(preds, actual)
    assert result["cum_s"][0]  == pytest.approx(1.0, abs=0.05)
    assert result["cum_bh"][0] == pytest.approx(1.0, abs=0.05)


# ── math_utils ────────────────────────────────────────────────────────────────

def test_safe_sharpe_zero_std():
    from utils.math_utils import safe_sharpe
    assert safe_sharpe(np.zeros(100)) == 0.0


def test_max_drawdown_flat():
    from utils.math_utils import max_drawdown
    assert max_drawdown(np.ones(50)) == 0.0


def test_max_drawdown_declining():
    from utils.math_utils import max_drawdown
    cum = np.array([1.0, 0.9, 0.8, 0.85, 0.7])
    assert max_drawdown(cum) < 0.0


# ── model factory ─────────────────────────────────────────────────────────────

def test_make_model_random_forest():
    from models.model_factory import make_model
    mdl = make_model("Random Forest")
    assert hasattr(mdl, "fit") and hasattr(mdl, "predict")


def test_make_model_linear_fits():
    from models.model_factory import make_model
    mdl = make_model("Linear Regression")
    X   = np.random.rand(50, 3)
    y   = np.random.rand(50)
    mdl.fit(X, y)
    assert len(mdl.predict(X)) == 50


def test_make_model_unknown_raises():
    from models.model_factory import make_model
    with pytest.raises(ValueError):
        make_model("SomeFakeModel")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
