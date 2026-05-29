"""
utils/helpers.py
================
Shared UI and math helpers used across all pages.
"""

import streamlit as st
import numpy as np


def page_header(title: str, subtitle: str = "", icon: str = "") -> None:
    """Render a consistent page header with title and optional subtitle."""
    st.markdown(
        f'<div style="margin-bottom:20px;">'
        f'<div style="font-size:1.75rem;font-weight:800;color:#f2ca50;letter-spacing:-0.01em;">'
        f'{icon + " " if icon else ""}{title}</div>'
        f'<div style="font-size:0.88rem;color:#d6e4f7;opacity:0.55;margin-top:3px;">{subtitle}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def disclaimer_banner(text: str | None = None) -> None:
    """Show a scientific disclaimer banner."""
    msg = text or (
        "<b style='color:#f2ca50;'>Research Disclaimer:</b> This tool estimates "
        "short-term market tendencies using historical & macro-financial indicators. "
        "Financial markets contain significant stochastic components — outputs should "
        "not be used as investment advice."
    )
    st.markdown(
        f'<div style="background:rgba(242,202,80,0.05);border:1px solid rgba(242,202,80,0.2);'
        f'border-left:3px solid #f2ca50;border-radius:8px;padding:8px 14px;'
        f'font-size:0.78rem;color:#d6e4f7;opacity:0.8;margin-bottom:16px;">'
        f'{msg}</div>',
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str = "",
             border_color: str = "#f2ca50") -> str:
    """Return HTML for a KPI card."""
    return (
        f'<div style="background:#13212e;border:1px solid rgba(255,255,255,0.08);'
        f'border-top:2px solid {border_color};border-radius:12px;padding:16px 18px;">'
        f'<div style="font-size:.7rem;color:#d6e4f7;opacity:.6;text-transform:uppercase;'
        f'letter-spacing:.05em;">{label}</div>'
        f'<div style="font-family:monospace;font-size:1.4rem;font-weight:700;color:{border_color};">'
        f'{value}</div>'
        f'<div style="font-size:.75rem;color:#d6e4f7;opacity:.55;margin-top:4px;">{sub}</div>'
        f'</div>'
    )


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
