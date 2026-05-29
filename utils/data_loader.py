"""
utils/data_loader.py
====================
Data loading, cleaning and preprocessing.
All functions are cached for Streamlit performance.
"""

import numpy as np
import pandas as pd
import streamlit as st


# ── Raw data ──────────────────────────────────────────────────────────────────

@st.cache_data
def load_data(csv_path: str = "TSDATA.csv") -> pd.DataFrame:
    """
    Load TSDATA.csv, clean currencies / volumes / percentages,
    remove flat/bad rows, forward-fill missing prices, and
    compute derived indicator columns (EMA, MACD, ATR).
    Also merges VIX + 10Y Treasury yield from yfinance.
    """
    df = pd.read_csv(csv_path)

    # ── Cleaners ─────────────────────────────────────────────────────────────
    def _clean_currency(x):
        if isinstance(x, str):
            return x.replace(",", "").replace("$", "").strip()
        return x

    def _clean_percent(x):
        if isinstance(x, str):
            x = x.replace("%", "").strip()
            try:
                return float(x) / 100
            except ValueError:
                return np.nan
        return x

    def _clean_volume(x):
        if isinstance(x, str):
            x = x.strip().upper()
            if x in ("", "-", "N/A", "NAN"):
                return np.nan
            try:
                if "K" in x:
                    return float(x.replace("K", "")) * 1_000
                elif "M" in x:
                    return float(x.replace("M", "")) * 1_000_000
                elif "B" in x:
                    return float(x.replace("B", "")) * 1_000_000_000
                return float(x.replace(",", ""))
            except ValueError:
                return np.nan
        return x

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    for col in ["Price_Gold", "High_Gold", "Low_Gold", "Open_Gold"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].apply(_clean_currency), errors="coerce")

    if "Change%_Gold" in df.columns:
        df["Change%_Gold"] = df["Change%_Gold"].apply(_clean_percent)

    if "Volume_Gold" in df.columns:
        df["Volume_Gold"] = df["Volume_Gold"].apply(_clean_volume)
        df["Volume_Gold"] = pd.to_numeric(df["Volume_Gold"], errors="coerce")

    for col in [
        "Price_Oil", "Price_Dollar", "High_Dollar", "Low_Dollar",
        "Open_Dollar", "Volume_Dollar", "Price_Stocks", "High_Stocks",
        "Low_Stocks", "Open_Stocks", "Volume_Stocks",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Preprocessing ─────────────────────────────────────────────────────────
    ohlc = ["Open_Gold", "High_Gold", "Low_Gold", "Price_Gold"]
    if all(c in df.columns for c in ohlc):
        flat = (
            (df["High_Gold"] == df["Low_Gold"])
            & (df["High_Gold"] == df["Open_Gold"])
            & (df["High_Gold"] == df["Price_Gold"])
        )
        df = df[~flat].reset_index(drop=True)

    if "Price_Oil" in df.columns:
        df["Price_Oil"] = df["Price_Oil"].clip(lower=0.1)

    price_cols = [
        "Price_Gold", "High_Gold", "Low_Gold", "Open_Gold",
        "Price_Oil", "Price_Dollar", "Price_Stocks",
    ]
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].ffill()

    # Cap extreme daily returns (>8%) to reduce outlier noise
    ret = df["Price_Gold"].pct_change() * 100
    clip_level = 8.0
    extreme = ret.abs() > clip_level
    if extreme.any():
        for i in df.index[extreme]:
            if i == 0:
                continue
            prev = df.loc[i - 1, "Price_Gold"]
            capped = np.clip(ret.loc[i], -clip_level, clip_level) / 100
            df.loc[i, "Price_Gold"] = prev * (1 + capped)

    df["Year"]  = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    # ── Economic data (VIX + 10Y Treasury) ──────────────────────────────────
    try:
        import yfinance as yf
        _eco = yf.download(
            ["^VIX", "^TNX"], start="1986-01-01",
            auto_adjust=False, progress=False,
        )["Close"]
        _eco.columns = ["TNX_Yield", "VIX"]
        _eco = _eco.reset_index()
        _eco["Date"] = pd.to_datetime(_eco["Date"]).dt.tz_localize(None)
        df = df.merge(_eco, on="Date", how="left")
        df["VIX"]       = df["VIX"].ffill()
        df["TNX_Yield"] = df["TNX_Yield"].ffill()
    except Exception:
        df["VIX"]       = np.nan
        df["TNX_Yield"] = np.nan

    # ── Technical indicators (price-normalised, stationary) ──────────────────
    p = df["Price_Gold"]
    df["EMA10_pct"] = (p.ewm(span=10, adjust=False).mean() / p - 1) * 100
    df["EMA20_pct"] = (p.ewm(span=20, adjust=False).mean() / p - 1) * 100
    ema12 = p.ewm(span=12, adjust=False).mean()
    ema26 = p.ewm(span=26, adjust=False).mean()
    df["MACD_pct"]  = (ema12 - ema26) / p * 100

    if all(c in df.columns for c in ["High_Gold", "Low_Gold"]):
        hl  = df["High_Gold"] - df["Low_Gold"]
        hpc = (df["High_Gold"] - p.shift(1)).abs()
        lpc = (df["Low_Gold"]  - p.shift(1)).abs()
        tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
        df["ATR_pct"] = tr.rolling(14).mean() / p * 100

    return df


@st.cache_data(ttl=3600)
def load_economic_data() -> pd.DataFrame:
    """Fetch VIX and 10Y Treasury Yield from yfinance — cached 1 hour."""
    try:
        import yfinance as yf
        eco = yf.download(
            ["^VIX", "^TNX"], period="max",
            auto_adjust=False, progress=False,
        )["Close"]
        eco.columns = ["TNX_Yield", "VIX"]
        eco = eco.reset_index().rename(columns={"Date": "Date"})
        eco["Date"] = pd.to_datetime(eco["Date"])
        return eco
    except Exception:
        return pd.DataFrame()
