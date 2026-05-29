"""
utils/math_utils.py
===================
Pure-Python / NumPy functions — no Streamlit dependency.
Safe to import in tests and scripts.
"""

import numpy as np


def safe_sharpe(returns: np.ndarray, periods: int = 252) -> float:
    """Annualised Sharpe ratio. Returns 0 if std is near-zero."""
    mu, sig = float(np.mean(returns)), float(np.std(returns))
    return float(mu / sig * np.sqrt(periods)) if sig > 1e-9 else 0.0


def safe_sortino(returns: np.ndarray, periods: int = 252) -> float:
    """Annualised Sortino ratio (downside deviation only)."""
    mu   = float(np.mean(returns))
    down = float(np.std(returns[returns < 0])) if np.any(returns < 0) else 1e-9
    return float(mu / down * np.sqrt(periods)) if down > 1e-9 else 0.0


def max_drawdown(cum_equity: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown as a percentage."""
    peak = np.maximum.accumulate(cum_equity)
    dd   = (cum_equity - peak) / np.where(peak > 1e-9, peak, 1e-9) * 100
    return float(dd.min())
