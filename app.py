import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Gold Price Prediction",
    page_icon="gold",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #061422; }
.block-container { padding-top: 1.5rem; max-width: 1600px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020f1c 0%, #0f1d2a 100%);
    border-right: 1px solid rgba(242,202,80,0.15);
}

/* ── Headings ── */
h1 { color: #f2ca50 !important; font-weight: 700; letter-spacing: -0.01em; }
h2, h3 { color: #f2ca50 !important; font-weight: 600; }

/* ── Metric Cards (Glassmorphism) ── */
[data-testid="stMetric"] {
    background: rgba(19,33,46,0.85);
    border: 1px solid rgba(255,255,255,0.08);
    border-top: 2px solid #f2ca50;
    border-radius: 12px;
    padding: 18px 20px;
    backdrop-filter: blur(12px);
    transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover { border-color: rgba(242,202,80,0.4); }
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.55rem !important;
    color: #f2ca50 !important;
    font-weight: 600;
}
[data-testid="stMetricLabel"] {
    font-size: 0.75rem; color: #d6e4f7;
    opacity: 0.6; text-transform: uppercase; letter-spacing: 0.05em;
}
[data-testid="stMetricDelta"] { font-size: 0.82rem; font-family: 'JetBrains Mono', monospace; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #f2ca50, #d4af37);
    color: #061422; font-weight: 700; font-family: 'Inter', sans-serif;
    border: none; border-radius: 8px; padding: 10px 28px;
    letter-spacing: 0.03em; transition: all 0.2s ease;
    box-shadow: 0 2px 12px rgba(242,202,80,0.2);
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(242,202,80,0.35);
    background: linear-gradient(135deg, #f5d060, #e8c040);
}

/* ── Inputs & Controls ── */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stMultiSelect"] label,
[data-testid="stNumberInput"] label {
    color: #d6e4f7 !important; font-size: 0.82rem;
    text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.7;
}
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: #13212e !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important; color: #d6e4f7 !important;
}
[data-testid="stNumberInput"] input {
    font-family: 'JetBrains Mono', monospace !important;
    background: #13212e !important; color: #f2ca50 !important;
    border: 1px solid rgba(242,202,80,0.2) !important; border-radius: 8px !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: rgba(19,33,46,0.7);
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px; backdrop-filter: blur(8px);
}
[data-testid="stExpander"] summary { color: #d6e4f7 !important; }

/* ── Dataframe / Tables ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; overflow: hidden;
}

/* ── Radio / nav ── */
.stRadio label { color: #d6e4f7 !important; font-size: 0.9rem; }
.stRadio [data-testid="stMarkdownContainer"] p { color: #d6e4f7 !important; }

/* ── Dividers ── */
hr { border-color: rgba(242,202,80,0.12) !important; }

/* ── Alerts ── */
[data-testid="stAlert"] { border-radius: 10px; }
[data-testid="stSuccess"] { background: rgba(34,197,94,0.1) !important; border-left: 3px solid #22c55e !important; }
[data-testid="stInfo"]    { background: rgba(96,165,250,0.1) !important; border-left: 3px solid #60a5fa !important; }
[data-testid="stError"]   { background: rgba(239,68,68,0.1)  !important; border-left: 3px solid #ef4444  !important; }

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    border: 1px solid rgba(242,202,80,0.4) !important;
    color: #f2ca50 !important; border-radius: 8px !important;
    font-weight: 500 !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(242,202,80,0.08) !important;
    border-color: #f2ca50 !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #f2ca50 !important; }

/* ── Slider track ── */
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stThumbValue"] { color: #f2ca50 !important; }
</style>
""", unsafe_allow_html=True)


def page_header(title, subtitle, icon=""):
    """Consistent page header with title + subtitle."""
    st.markdown(
        f'<div style="margin-bottom:20px;">'
        f'<div style="font-size:1.75rem;font-weight:800;color:#f2ca50;letter-spacing:-0.01em;">{icon} {title}</div>'
        f'<div style="font-size:0.88rem;color:#d6e4f7;opacity:0.55;margin-top:3px;">{subtitle}</div>'
        f'</div>',
        unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_economic_data():
    """Fetch VIX and 10Y Treasury Yield from yfinance — cached 1 hour."""
    try:
        import yfinance as yf
        eco = yf.download(["^VIX","^TNX"], period="max",
                           auto_adjust=False, progress=False)["Close"]
        eco.columns = ["TNX_Yield", "VIX"]
        eco = eco.reset_index().rename(columns={"Date":"Date"})
        eco["Date"] = pd.to_datetime(eco["Date"])
        return eco
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_data():
    df = pd.read_csv("TSDATA.csv")

    def clean_currency(x):
        if isinstance(x, str):
            return x.replace(',', '').replace('$', '').strip()
        return x

    def clean_percent(x):
        if isinstance(x, str):
            x = x.replace('%', '').strip()
            try:
                return float(x) / 100
            except ValueError:
                return np.nan
        return x

    def clean_volume(x):
        if isinstance(x, str):
            x = x.strip().upper()
            if x in ['', '-', 'N/A', 'NAN']:
                return np.nan
            try:
                if 'K' in x:
                    return float(x.replace('K', '')) * 1_000
                elif 'M' in x:
                    return float(x.replace('M', '')) * 1_000_000
                elif 'B' in x:
                    return float(x.replace('B', '')) * 1_000_000_000
                else:
                    return float(x.replace(',', ''))
            except ValueError:
                return np.nan
        return x

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

    for col in ['Price_Gold', 'High_Gold', 'Low_Gold', 'Open_Gold']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].apply(clean_currency), errors='coerce')

    if 'Change%_Gold' in df.columns:
        df['Change%_Gold'] = df['Change%_Gold'].apply(clean_percent)

    if 'Volume_Gold' in df.columns:
        df['Volume_Gold'] = df['Volume_Gold'].apply(clean_volume)
        df['Volume_Gold'] = pd.to_numeric(df['Volume_Gold'], errors='coerce')

    for col in ['Price_Oil', 'Price_Dollar', 'High_Dollar', 'Low_Dollar',
                'Open_Dollar', 'Volume_Dollar', 'Price_Stocks', 'High_Stocks',
                'Low_Stocks', 'Open_Stocks', 'Volume_Stocks']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # ── Preprocessing ────────────────────────────────────────────────────────

    # 1. Remove flat/bad rows: High == Low == Open == Close (no real trading data)
    ohlc = ['Open_Gold', 'High_Gold', 'Low_Gold', 'Price_Gold']
    if all(c in df.columns for c in ohlc):
        flat_mask = (
            (df['High_Gold'] == df['Low_Gold']) &
            (df['High_Gold'] == df['Open_Gold']) &
            (df['High_Gold'] == df['Price_Gold'])
        )
        df = df[~flat_mask].reset_index(drop=True)

    # 2. Clip negative Oil prices (April 2020 anomaly — keep the event but clip)
    if 'Price_Oil' in df.columns:
        df['Price_Oil'] = df['Price_Oil'].clip(lower=0.1)

    # 3. Forward-fill missing prices (carries last valid price forward)
    price_cols = ['Price_Gold', 'High_Gold', 'Low_Gold', 'Open_Gold',
                  'Price_Oil', 'Price_Dollar', 'Price_Stocks']
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].ffill()

    # 4. Clip extreme daily gold returns (beyond ±8%) — reduces outlier noise
    #    The -11% in Jan 2026 is a real event but extremely rare; cap at 3σ level
    ret = df['Price_Gold'].pct_change() * 100
    clip_level = 8.0
    extreme = ret.abs() > clip_level
    if extreme.any():
        # Adjust prices to respect the cap (work backwards from clipped returns)
        for i in df.index[extreme]:
            if i == 0:
                continue
            prev_price = df.loc[i - 1, 'Price_Gold']
            actual_ret = ret.loc[i]
            capped_ret = np.clip(actual_ret, -clip_level, clip_level) / 100
            df.loc[i, 'Price_Gold'] = prev_price * (1 + capped_ret)

    # ── Merge Economic Data (VIX + 10Y Treasury) ─────────────────────────
    try:
        import yfinance as yf
        _eco = yf.download(["^VIX","^TNX"], start="1986-01-01",
                            auto_adjust=False, progress=False)["Close"]
        _eco.columns = ["TNX_Yield","VIX"]
        _eco = _eco.reset_index()
        _eco["Date"] = pd.to_datetime(_eco["Date"]).dt.tz_localize(None)
        df = df.merge(_eco, on="Date", how="left")
        df["VIX"]       = df["VIX"].ffill()
        df["TNX_Yield"] = df["TNX_Yield"].ffill()
    except Exception:
        df["VIX"]       = np.nan
        df["TNX_Yield"] = np.nan

    df['Year']  = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    # Derived indicator features (price-normalised, stationary)
    p = df['Price_Gold']
    df['EMA10_pct'] = (p.ewm(span=10, adjust=False).mean() / p - 1) * 100
    df['EMA20_pct'] = (p.ewm(span=20, adjust=False).mean() / p - 1) * 100
    ema12 = p.ewm(span=12, adjust=False).mean()
    ema26 = p.ewm(span=26, adjust=False).mean()
    df['MACD_pct']  = (ema12 - ema26) / p * 100
    if all(c in df.columns for c in ['High_Gold', 'Low_Gold']):
        hl  = df['High_Gold'] - df['Low_Gold']
        hpc = (df['High_Gold'] - p.shift(1)).abs()
        lpc = (df['Low_Gold']  - p.shift(1)).abs()
        tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
        df['ATR_pct'] = tr.rolling(14).mean() / p * 100

    return df


df = load_data()

# ── Sidebar ────────────────────────────────────────────────────────────────────
# ── Add nav button CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Sidebar nav styling ── */
div[data-testid="stRadio"] > div { gap: 2px !important; }

/* Each nav item label */
div[data-testid="stRadio"] label {
    display: flex !important;
    align-items: center !important;
    padding: 9px 14px !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: background 0.15s !important;
    border: 1px solid transparent !important;
    width: 100% !important;
}
div[data-testid="stRadio"] label:hover {
    background: rgba(242,202,80,0.07) !important;
    border-color: rgba(242,202,80,0.15) !important;
}

/* Hide just the radio circle indicator — keep text visible */
div[data-testid="stRadio"] label > div:first-child {
    display: none !important;
}

/* Text inside the label */
div[data-testid="stRadio"] label p,
div[data-testid="stRadio"] label span {
    color: #d6e4f7 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    margin: 0 !important;
}

/* Active item — Streamlit marks selected with aria-checked or input:checked */
div[data-testid="stRadio"] label:has(input:checked) {
    background: rgba(242,202,80,0.12) !important;
    border-color: rgba(242,202,80,0.35) !important;
}
div[data-testid="stRadio"] label:has(input:checked) p,
div[data-testid="stRadio"] label:has(input:checked) span {
    color: #f2ca50 !important;
    font-weight: 600 !important;
}

/* Date inputs */
[data-testid="stDateInput"] input {
    background: #13212e !important; color: #d6e4f7 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    font-family: "JetBrains Mono", monospace !important;
    font-size: 0.82rem !important;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    # ── Logo ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding:20px 0 12px 0;">
        <svg viewBox="0 0 120 120" width="72" height="72" xmlns="http://www.w3.org/2000/svg"
             style="display:block;margin:0 auto 10px auto;">
          <defs>
            <radialGradient id="cg" cx="38%" cy="35%" r="65%">
              <stop offset="0%" stop-color="#ffe566"/>
              <stop offset="55%" stop-color="#f2ca50"/>
              <stop offset="100%" stop-color="#b8860b"/>
            </radialGradient>
            <radialGradient id="eg" cx="40%" cy="38%" r="60%">
              <stop offset="0%" stop-color="#ffd700"/>
              <stop offset="100%" stop-color="#b8860b"/>
            </radialGradient>
          </defs>
          <circle cx="60" cy="60" r="58" fill="#b8860b"/>
          <circle cx="60" cy="60" r="54" fill="url(#cg)"/>
          <circle cx="60" cy="60" r="50" fill="none" stroke="#e6a800" stroke-width="1.5" opacity="0.6"/>
          <circle cx="60" cy="60" r="46" fill="url(#eg)"/>
          <text x="63" y="80" text-anchor="middle" font-family="Georgia,serif" font-size="54"
                font-weight="900" fill="#b8860b" opacity="0.45">$</text>
          <text x="60" y="78" text-anchor="middle" font-family="Georgia,serif" font-size="54"
                font-weight="900" fill="#f2ca50">$</text>
          <ellipse cx="44" cy="38" rx="12" ry="6" fill="white" opacity="0.18" transform="rotate(-30 44 38)"/>
        </svg>
        <div style="font-size:1.1rem;font-weight:800;color:#f2ca50;letter-spacing:1px;">AURUM</div>
        <div style="font-size:0.68rem;color:#d6e4f7;opacity:0.5;letter-spacing:3px;margin-top:2px;">GOLD INTELLIGENCE</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(242,202,80,0.1);margin:0 0 10px 0;'>", unsafe_allow_html=True)

    # ── Navigation ────────────────────────────────────────────────────────────
    NAV_ITEMS = {
        "Home":       ("🏠", "Welcome"),
        "Dashboard":  ("📊", "Charts & Analysis"),
        "Prediction": ("🤖", "Train ML Models"),
        "Forecast":   ("🔮", "30-Day Outlook"),
        "Simulator":  ("💰", "Investment Calc"),
        "Portfolio":  ("📐", "Optimize Allocation"),
        "Sentiment":  ("📰", "News Sentiment"),
        "About":      ("ℹ️",  "App Info"),
    }
    page = st.radio(
        "nav",
        list(NAV_ITEMS.keys()),
        format_func=lambda p: f"{NAV_ITEMS[p][0]}  {p}",
        label_visibility="collapsed")

    st.markdown("<hr style='border-color:rgba(242,202,80,0.1);margin:10px 0;'>", unsafe_allow_html=True)

    # ── Page-specific controls ────────────────────────────────────────────────
    if page == "Dashboard":
        st.markdown('<div style="font-size:.72rem;color:#d6e4f7;opacity:.5;text-transform:uppercase;letter-spacing:.08em;padding:0 2px 4px;">Price Unit</div>', unsafe_allow_html=True)
        unit = st.radio("unit", ["oz (Ounce)", "g (Gram)"],
                        horizontal=True, label_visibility="collapsed")
        st.markdown('<div style="font-size:.72rem;color:#d6e4f7;opacity:.5;text-transform:uppercase;letter-spacing:.08em;padding:6px 2px 4px;">Karat</div>', unsafe_allow_html=True)
        karat = st.selectbox("karat", ["24K (999 — Pure)", "22K (916)", "21K (875)", "18K (750)"],
                             label_visibility="collapsed")
        st.markdown("<hr style='border-color:rgba(242,202,80,0.1);margin:10px 0;'>", unsafe_allow_html=True)
    else:
        unit  = "oz (Ounce)"
        karat = "24K (999 — Pure)"

    if page not in ("About", "Forecast", "Home", "Simulator", "Sentiment", "Portfolio"):
        st.markdown('<div style="font-size:.72rem;color:#d6e4f7;opacity:.5;text-transform:uppercase;letter-spacing:.08em;padding:0 2px 4px;">Date Range</div>', unsafe_allow_html=True)
        min_date = df["Date"].min().date()
        max_date = df["Date"].max().date()
        start_date = st.date_input("From", value=pd.to_datetime("2010-01-01").date(),
                                   min_value=min_date, max_value=max_date,
                                   help="Start of analysis period")
        end_date   = st.date_input("To", value=max_date,
                                   min_value=min_date, max_value=max_date,
                                   help="End of analysis period")
        st.markdown("<hr style='border-color:rgba(242,202,80,0.1);margin:10px 0;'>", unsafe_allow_html=True)
    else:
        start_date = pd.to_datetime("2010-01-01").date()
        end_date   = df["Date"].max().date()

    # ── Data freshness badge ──────────────────────────────────────────────────
    _last_date  = df["Date"].max()
    _days_ago   = (pd.Timestamp.today() - _last_date).days
    _fresh_color = "#22c55e" if _days_ago <= 3 else "#f2ca50" if _days_ago <= 7 else "#ef4444"
    _fresh_txt   = "Live" if _days_ago <= 3 else f"{_days_ago}d ago"
    st.markdown(f"""
    <div style="background:#13212e;border:1px solid rgba(255,255,255,0.06);border-radius:8px;
                padding:8px 12px;font-size:0.75rem;color:#d6e4f7;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;">
            <span style="opacity:.5;text-transform:uppercase;letter-spacing:.05em;font-size:.65rem;">Data</span>
            <span style="color:{_fresh_color};font-weight:600;">{_fresh_txt}</span>
        </div>
        <div style="opacity:.6;">{_last_date.strftime('%b %d, %Y')} &nbsp;·&nbsp; {len(df):,} days</div>
        <div style="opacity:.4;font-size:.65rem;margin-top:2px;">Yahoo Finance · Auto-updated daily</div>
    </div>
    """, unsafe_allow_html=True)

mask        = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
filtered_df = df[mask].copy()


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "Home":
    import streamlit.components.v1 as components

    # GLB served via Streamlit static file serving
    # File lives in ./static/gold_coins.glb
    # Accessible at /app/static/gold_coins.glb on Streamlit Cloud
    _glb_url = "./app/static/gold_coins.glb"

    _html = """
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <script type="module"
      src="https://unpkg.com/@google/model-viewer@3.3.0/dist/model-viewer.min.js">
    </script>
    <style>
      * { margin:0; padding:0; box-sizing:border-box; }
      body { font-family: sans-serif; overflow:hidden; background:#061422; }
      .hero {
        display:flex; flex-direction:column; align-items:center;
        justify-content:center; height:100vh; width:100%;
        position: relative; z-index: 1;
      }
      model-viewer {
        width:420px; height:420px; background:transparent;
        --progress-bar-color: #f2ca50;
      }
      .title {
        font-size:2.8rem; font-weight:900; color:#f2ca50;
        letter-spacing:2px; text-align:center; margin-top:8px;
        text-shadow: 0 0 40px rgba(255,199,44,0.4);
      }
      .subtitle {
        font-size:1rem; color:#d6e4f7; opacity:0.6; margin-top:10px;
        letter-spacing:3px; text-align:center; text-transform:uppercase;
      }
      .badge {
        display:inline-block; margin-top:22px;
        background:rgba(255,199,44,0.12);
        border:1px solid rgba(255,199,44,0.35);
        color:#f2ca50; padding:8px 24px; border-radius:999px;
        font-size:0.8rem; letter-spacing:2px; text-transform:uppercase;
      }
    </style>
    </head>
    <body>
      <div class="hero">
        <model-viewer
          src="./app/static/gold_coins.glb"
          auto-rotate
          auto-rotate-delay="0"
          rotation-per-second="20deg"
          camera-controls
          disable-zoom
          shadow-intensity="1.2"
          exposure="1.1"
          environment-image="neutral"
        ></model-viewer>
        <div class="title">Gold Price Prediction</div>
        <div class="subtitle">AI-Powered Market Analysis</div>
        <div class="badge">Data from Yahoo Finance &nbsp;&middot;&nbsp; Updated Daily</div>
      </div>
    </body>
    </html>
    """

    components.html(_html, height=640, scrolling=False)

    st.markdown('''
    <div style="text-align:center;color:#d6e4f7;opacity:0.45;font-size:0.78rem;margin-top:4px;">
        Drag to rotate &nbsp;&middot;&nbsp; Use the sidebar to navigate
    </div>
    ''', unsafe_allow_html=True)


if page == "Dashboard":

    # Unit conversion setup
    _purity_map = {"24K (999 — Pure)": 0.999, "22K (916)": 0.916,
                   "21K (875)": 0.875, "18K (750)": 0.750}
    _purity = _purity_map.get(karat, 1.0)
    _knum   = karat[:3]  # "24K", "22K", etc.
    _pdiv   = (31.1035 if unit == "g (Gram)" else 1.0) / _purity
    _ulbl   = "g" if unit == "g (Gram)" else "oz"
    _ufmt   = lambda v: f"${v:,.3f}" if _ulbl == "g" else f"${v:,.2f}"
    disp_df = filtered_df.copy()
    for _c in ['Price_Gold', 'High_Gold', 'Low_Gold', 'Open_Gold']:
        if _c in disp_df.columns:
            disp_df[_c] = disp_df[_c] / _pdiv

    # Hero image + title
    hero_col, title_col = st.columns([1, 2])
    with hero_col:
        st.image("gold_hero.jpg", use_container_width=True)
    with title_col:
        page_header("Gold Price Dashboard",
                    f"{start_date} → {end_date}  ·  {len(filtered_df):,} trading days")

    # KPI Cards
    gold_clean = disp_df.dropna(subset=['Price_Gold'])
    if not gold_clean.empty:
        latest = gold_clean.iloc[-1]
        first  = gold_clean.iloc[0]
        delta  = latest['Price_Gold'] - first['Price_Gold']
        pct    = (delta / first['Price_Gold']) * 100
        c1, c2, c3, c4 = st.columns(4)
        _d_str = f"{delta:+,.3f}" if _ulbl=="g" else f"{delta:+,.2f}"
        c1.metric(f"Current Price — {_knum} ({_ulbl})",  _ufmt(latest['Price_Gold']), f"{_d_str} ({pct:+.1f}%)")
        c2.metric("Period High",    _ufmt(gold_clean['High_Gold'].max()))
        c3.metric("Period Low",     _ufmt(gold_clean['Low_Gold'].min()))
        c4.metric("Average Price",  _ufmt(gold_clean['Price_Gold'].mean()))

    st.markdown("---")

    # Price Chart
    chart_col, ctrl_col = st.columns([4, 1])
    with chart_col:
        st.subheader("Gold Price Over Time")
    with ctrl_col:
        st.markdown("<div style='padding-top:28px'></div>", unsafe_allow_html=True)
        chart_type = st.radio("ct", ["Candlestick", "Line"],
                              horizontal=True, label_visibility="collapsed")

    candle_df = disp_df.dropna(subset=['Open_Gold', 'High_Gold', 'Low_Gold', 'Price_Gold'])
    fig = go.Figure()

    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=candle_df['Date'],
            open=candle_df['Open_Gold'], high=candle_df['High_Gold'],
            low=candle_df['Low_Gold'],   close=candle_df['Price_Gold'],
            name='OHLC',
            increasing=dict(line=dict(color='#f2ca50', width=1), fillcolor='rgba(255,199,44,0.85)'),
            decreasing=dict(line=dict(color='#ef4444', width=1), fillcolor='rgba(239,68,68,0.75)'),
        ))
        if 'Volume_Gold' in candle_df.columns:
            vol_df   = candle_df.dropna(subset=['Volume_Gold'])
            colors_v = ['rgba(255,199,44,0.4)' if c >= o else 'rgba(239,68,68,0.35)'
                        for c, o in zip(vol_df['Price_Gold'], vol_df['Open_Gold'])]
            fig.add_trace(go.Bar(x=vol_df['Date'], y=vol_df['Volume_Gold'],
                                 name='Volume', marker_color=colors_v,
                                 yaxis='y2', showlegend=False))
        fig.update_layout(
            yaxis2=dict(overlaying='y', side='right', showgrid=False, showticklabels=False,
                        range=[0, candle_df['Volume_Gold'].max() * 5]
                        if 'Volume_Gold' in candle_df.columns else {}),
            xaxis_rangeslider_visible=False,
        )
    else:
        fig.add_trace(go.Scatter(x=disp_df['Date'], y=disp_df['Price_Gold'],
                                  mode='lines', name='Close',
                                  line=dict(color='#f2ca50', width=2),
                                  fill='tozeroy', fillcolor='rgba(255,199,44,0.08)'))
        fig.add_trace(go.Scatter(x=disp_df['Date'], y=disp_df['High_Gold'],
                                  mode='lines', name='High',
                                  line=dict(color='rgba(0,200,0,0.4)', width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=disp_df['Date'], y=disp_df['Low_Gold'],
                                  mode='lines', name='Low',
                                  line=dict(color='rgba(255,80,80,0.4)', width=1, dash='dot'),
                                  fill='tonexty', fillcolor='rgba(180,180,180,0.04)'))

    fig.update_layout(height=480, template='plotly_dark',
                      paper_bgcolor='#0d1b2a', plot_bgcolor='#061422',
                      xaxis_title="Date", yaxis_title=f"Price (USD/{_ulbl})",
                      hovermode='x unified',
                      legend=dict(orientation='h', y=1.02, x=0),
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width='stretch')

    # ── Technical Indicators ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Technical Indicators")

    ti_c1, ti_c2 = st.columns(2)
    with ti_c1:
        bb_period = st.slider("Bollinger Bands Period", 10, 50, 20, key="bb_p")
    with ti_c2:
        rsi_period = st.slider("RSI Period", 7, 21, 14, key="rsi_p")

    tech_df = disp_df[['Date', 'Price_Gold']].dropna().copy().reset_index(drop=True)
    tech_df['SMA']      = tech_df['Price_Gold'].rolling(bb_period).mean()
    tech_df['STD']      = tech_df['Price_Gold'].rolling(bb_period).std()
    tech_df['BB_upper'] = tech_df['SMA'] + 2 * tech_df['STD']
    tech_df['BB_lower'] = tech_df['SMA'] - 2 * tech_df['STD']

    delta_r = tech_df['Price_Gold'].diff()
    gain    = delta_r.clip(lower=0).rolling(rsi_period).mean()
    loss    = (-delta_r.clip(upper=0)).rolling(rsi_period).mean()
    rs      = gain / loss.replace(0, np.nan)
    tech_df['RSI'] = 100 - (100 / (1 + rs))

    rsi_valid   = tech_df['RSI'].dropna()
    current_rsi = rsi_valid.iloc[-1] if not rsi_valid.empty else None
    latest_upper = tech_df['BB_upper'].iloc[-1]
    latest_lower = tech_df['BB_lower'].iloc[-1]
    latest_sma   = tech_df['SMA'].iloc[-1]

    if current_rsi is not None:
        if current_rsi >= 70:
            rsi_label = "Overbought"
        elif current_rsi <= 30:
            rsi_label = "Oversold"
        else:
            rsi_label = "Neutral"
        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("RSI",         f"{current_rsi:.1f}", rsi_label)
        sm2.metric("SMA (BB mid)",_ufmt(latest_sma)   if not np.isnan(latest_sma)   else "—")
        sm3.metric("Upper Band",  _ufmt(latest_upper) if not np.isnan(latest_upper) else "—")
        sm4.metric("Lower Band",  _ufmt(latest_lower) if not np.isnan(latest_lower) else "—")

    bb_col, rsi_col = st.columns(2)

    with bb_col:
        st.markdown("**Bollinger Bands**")
        bb_plot = tech_df.dropna(subset=['SMA'])
        fig_bb  = go.Figure()
        fig_bb.add_trace(go.Scatter(x=bb_plot['Date'], y=bb_plot['BB_upper'],
                                     line=dict(color='rgba(46,95,101,0)', width=0),
                                     showlegend=False, name='Upper'))
        fig_bb.add_trace(go.Scatter(x=bb_plot['Date'], y=bb_plot['BB_lower'],
                                     fill='tonexty', fillcolor='rgba(46,95,101,0.18)',
                                     line=dict(color='rgba(46,95,101,0)', width=0),
                                     showlegend=False, name='Lower'))
        fig_bb.add_trace(go.Scatter(x=bb_plot['Date'], y=bb_plot['BB_upper'],
                                     line=dict(color='#293644', width=1.2, dash='dot'),
                                     name='Upper Band'))
        fig_bb.add_trace(go.Scatter(x=bb_plot['Date'], y=bb_plot['BB_lower'],
                                     line=dict(color='#293644', width=1.2, dash='dot'),
                                     name='Lower Band'))
        fig_bb.add_trace(go.Scatter(x=bb_plot['Date'], y=bb_plot['SMA'],
                                     line=dict(color='#e0f7fa', width=1.5, dash='dash'),
                                     name=f'SMA {bb_period}'))
        fig_bb.add_trace(go.Scatter(x=bb_plot['Date'], y=bb_plot['Price_Gold'],
                                     line=dict(color='#f2ca50', width=2),
                                     name='Gold Price'))
        fig_bb.update_layout(height=360, template='plotly_dark',
                              paper_bgcolor='#0d1b2a', plot_bgcolor='#061422',
                              yaxis_title=f"Price (USD/{_ulbl})", hovermode='x unified',
                              legend=dict(orientation='h', y=1.02, font=dict(size=10)),
                              margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_bb, width='stretch')

    with rsi_col:
        st.markdown("**Relative Strength Index (RSI)**")
        rsi_plot = tech_df.dropna(subset=['RSI'])
        fig_rsi  = go.Figure()
        fig_rsi.add_hrect(y0=70, y1=100, fillcolor='rgba(239,68,68,0.08)', line_width=0,
                           annotation_text="Overbought", annotation_position="top right",
                           annotation=dict(font_size=10, font_color='#ef4444'))
        fig_rsi.add_hrect(y0=0, y1=30, fillcolor='rgba(34,197,94,0.08)', line_width=0,
                           annotation_text="Oversold", annotation_position="bottom right",
                           annotation=dict(font_size=10, font_color='#22c55e'))
        fig_rsi.add_hline(y=70, line=dict(color='#ef4444', width=1, dash='dot'))
        fig_rsi.add_hline(y=30, line=dict(color='#22c55e', width=1, dash='dot'))
        fig_rsi.add_hline(y=50, line=dict(color='#4a5568', width=1, dash='dot'))
        fig_rsi.add_trace(go.Scatter(x=rsi_plot['Date'], y=rsi_plot['RSI'],
                                      mode='lines', name=f'RSI ({rsi_period})',
                                      line=dict(color='#f2ca50', width=2)))
        fig_rsi.update_layout(height=360, template='plotly_dark',
                               paper_bgcolor='#0d1b2a', plot_bgcolor='#061422',
                               yaxis=dict(title="RSI", range=[0, 100],
                                          tickvals=[0, 30, 50, 70, 100]),
                               hovermode='x unified',
                               legend=dict(orientation='h', y=1.02, font=dict(size=10)),
                               margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_rsi, width='stretch')

    st.markdown("---")

    # ── Asset Comparison & Correlation ────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.subheader("Asset Comparison (Normalized)")
        cmp = filtered_df[['Date', 'Price_Gold', 'Price_Oil',
                            'Price_Dollar', 'Price_Stocks']].dropna()
        if len(cmp) > 1:
            vals = MinMaxScaler((0, 100)).fit_transform(
                cmp[['Price_Gold', 'Price_Oil', 'Price_Dollar', 'Price_Stocks']])
            norm = pd.DataFrame(vals, columns=['Gold', 'Oil', 'Dollar Index', 'S&P 500'])
            norm['Date'] = cmp['Date'].values
            fig2    = go.Figure()
            palette = {'Gold': '#f2ca50', 'Oil': '#1e2b39',
                       'Dollar Index': '#e0f7fa', 'S&P 500': '#4a5568'}
            for col, color in palette.items():
                fig2.add_trace(go.Scatter(x=norm['Date'], y=norm[col],
                                           mode='lines', name=col,
                                           line=dict(color=color, width=1.5)))
            fig2.update_layout(height=340, template='plotly_dark',
                               paper_bgcolor='#0d1b2a', plot_bgcolor='#061422',
                               yaxis_title="Normalized (0–100)", hovermode='x unified',
                               legend=dict(orientation='h', y=1.02))
            st.plotly_chart(fig2, width='stretch')
        else:
            st.info("Not enough data for comparison.")

    with right:
        st.subheader("Correlation Heatmap")
        corr_df = filtered_df[['Price_Gold', 'Price_Oil',
                                'Price_Dollar', 'Price_Stocks']].dropna()
        if len(corr_df) > 1:
            fig3 = px.imshow(corr_df.corr(), text_auto='.2f',
                              color_continuous_scale='RdYlGn', zmin=-1, zmax=1)
            fig3.update_layout(height=340, template='plotly_dark',
                               paper_bgcolor='#0d1b2a', plot_bgcolor='#061422')
            st.plotly_chart(fig3, width='stretch')
        else:
            st.info("Not enough data for correlation.")

    # ── Yearly Average & Volume ───────────────────────────────────────────────
    l2, r2 = st.columns(2)

    with l2:
        st.subheader("Yearly Average Gold Price")
        yearly = disp_df.groupby('Year')['Price_Gold'].mean().reset_index()
        fig4 = px.bar(yearly, x='Year', y='Price_Gold', color='Price_Gold',
                      color_continuous_scale=[[0, '#1e2b39'], [0.5, '#f2ca50'], [1, '#ffffff']],
                      labels={'Price_Gold': f'Avg Price (USD/{_ulbl})'})
        fig4.update_layout(height=300, template='plotly_dark',
                           paper_bgcolor='#0d1b2a', plot_bgcolor='#061422',
                           showlegend=False, xaxis_title="")
        st.plotly_chart(fig4, width='stretch')

    with r2:
        st.subheader("Trading Volume")
        vol = filtered_df[['Date', 'Volume_Gold']].dropna()
        if not vol.empty:
            fig5 = go.Figure(go.Bar(x=vol['Date'], y=vol['Volume_Gold'],
                                     marker_color='rgba(255,199,44,0.7)'))
            fig5.update_layout(height=300, template='plotly_dark',
                               paper_bgcolor='#0d1b2a', plot_bgcolor='#061422',
                               yaxis_title="Volume", xaxis_title="")
            st.plotly_chart(fig5, width='stretch')
        else:
            st.info("No volume data in selected range.")

    # ── Daily Change Distribution ─────────────────────────────────────────────
    st.subheader("Daily Change % Distribution")
    chg = filtered_df['Change%_Gold'].dropna() * 100
    if not chg.empty:
        fig6 = px.histogram(chg, nbins=80, color_discrete_sequence=['#f2ca50'],
                             labels={'value': 'Daily Change (%)', 'count': 'Days'})
        fig6.update_layout(height=250, template='plotly_dark',
                           paper_bgcolor='#0d1b2a', plot_bgcolor='#061422',
                           xaxis_title="Daily Change (%)", showlegend=False)
        st.plotly_chart(fig6, width='stretch')

    # ── MACD Chart ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("MACD (Moving Average Convergence Divergence)")
    macd_df = disp_df[['Date', 'Price_Gold']].dropna().copy().reset_index(drop=True)
    _p       = macd_df['Price_Gold']
    _ema12   = _p.ewm(span=12, adjust=False).mean()
    _ema26   = _p.ewm(span=26, adjust=False).mean()
    macd_df['MACD']   = _ema12 - _ema26
    macd_df['Signal'] = macd_df['MACD'].ewm(span=9, adjust=False).mean()
    macd_df['Hist']   = macd_df['MACD'] - macd_df['Signal']
    macd_df = macd_df.dropna()

    _macd_latest = macd_df['MACD'].iloc[-1]
    _sig_latest  = macd_df['Signal'].iloc[-1]
    _cross_txt   = "Bullish crossover" if _macd_latest > _sig_latest else "Bearish crossover"
    _cross_color = "#22c55e" if _macd_latest > _sig_latest else "#ef4444"
    mq1, mq2, mq3 = st.columns(3)
    mq1.metric("MACD Line",   f"{_macd_latest:.4f}")
    mq2.metric("Signal Line", f"{_sig_latest:.4f}")
    mq3.metric("Status", _cross_txt)

    fig_macd = go.Figure()
    _colors_hist = ['#22c55e' if v >= 0 else '#ef4444' for v in macd_df['Hist']]
    fig_macd.add_trace(go.Bar(
        x=macd_df['Date'], y=macd_df['Hist'],
        name='Histogram', marker_color=_colors_hist,
        opacity=0.7, showlegend=True))
    fig_macd.add_trace(go.Scatter(
        x=macd_df['Date'], y=macd_df['MACD'],
        name='MACD', line=dict(color='#f2ca50', width=2)))
    fig_macd.add_trace(go.Scatter(
        x=macd_df['Date'], y=macd_df['Signal'],
        name='Signal', line=dict(color='#e0f7fa', width=1.5, dash='dot')))
    fig_macd.add_hline(y=0, line=dict(color='#4a5568', width=1, dash='dot'))
    fig_macd.update_layout(
        height=320, template='plotly_dark',
        paper_bgcolor='#0d1b2a', plot_bgcolor='#061422',
        yaxis_title="MACD Value", hovermode='x unified',
        legend=dict(orientation='h', y=1.02, font=dict(size=10)),
        margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_macd, width='stretch')

    # ── Market Regime Detection ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Market Regime Detection")
    st.markdown(
        '<div style="font-size:.82rem;color:#d6e4f7;opacity:.6;margin-bottom:12px;">'
        'Classifies the current market state using rolling volatility, 60-day momentum and SMA-200.</div>',
        unsafe_allow_html=True)

    _reg_df = filtered_df[["Date","Price_Gold"]].dropna().copy().reset_index(drop=True)
    _reg_df["ret"]     = _reg_df["Price_Gold"].pct_change() * 100
    _reg_df["vol_21"]  = _reg_df["ret"].rolling(21).std() * np.sqrt(252)
    _reg_df["mom_60"]  = _reg_df["Price_Gold"].pct_change(60) * 100
    _reg_df["sma_200"] = _reg_df["Price_Gold"].rolling(200).mean()
    _reg_df = _reg_df.dropna().reset_index(drop=True)

    if not _reg_df.empty:
        _vol_hi = _reg_df["vol_21"].quantile(0.75)
        _vol_lo = _reg_df["vol_21"].quantile(0.35)

        def _classify(row):
            if row["vol_21"] >= _vol_hi:                                     return "High Volatility"
            elif row["Price_Gold"] > row["sma_200"] and row["mom_60"] > 3:  return "Bull Market"
            elif row["Price_Gold"] < row["sma_200"] and row["mom_60"] < -3: return "Bear Market"
            elif row["vol_21"] <= _vol_lo and row["mom_60"] > 0:            return "Quiet Bull"
            elif row["vol_21"] <= _vol_lo:                                   return "Quiet Bear"
            else:                                                             return "Transition"

        _reg_df["regime"] = _reg_df.apply(_classify, axis=1)
        _reg_colors = {"Bull Market":"#22c55e","Bear Market":"#ef4444",
                       "High Volatility":"#f2ca50","Quiet Bull":"#86efac",
                       "Quiet Bear":"#fca5a5","Transition":"#94a3b8"}

        _cur  = _reg_df["regime"].iloc[-1]
        _ccol = _reg_colors.get(_cur, "#94a3b8")
        _cvol = _reg_df["vol_21"].iloc[-1]
        _cmom = _reg_df["mom_60"].iloc[-1]
        _csma = (_reg_df["Price_Gold"].iloc[-1] / _reg_df["sma_200"].iloc[-1] - 1) * 100
        _rdist = _reg_df["regime"].value_counts(normalize=True) * 100

        rg1, rg2, rg3, rg4 = st.columns(4)
        rg1.markdown(
            f'<div style="background:#13212e;border:1px solid rgba(255,255,255,0.08);'
            f'border-top:2px solid {_ccol};border-radius:12px;padding:16px 18px;">'
            f'<div style="font-size:.7rem;color:#d6e4f7;opacity:.6;text-transform:uppercase;letter-spacing:.05em;">Current Regime</div>'
            f'<div style="font-size:1.4rem;font-weight:800;color:{_ccol};">{_cur}</div>'
            f'<div style="font-size:.75rem;color:#d6e4f7;opacity:.55;margin-top:4px;">Vol: {_cvol:.1f}% · Mom60: {_cmom:+.1f}%</div></div>',
            unsafe_allow_html=True)
        rg2.metric("Ann. Volatility (21d)", f"{_cvol:.1f}%", help="Rolling 21-day vol, annualised")
        rg3.metric("Momentum (60d)",        f"{_cmom:+.1f}%", help="% change over last 60 trading days")
        rg4.metric("vs SMA-200",            f"{_csma:+.1f}%", help="Current price vs 200-day moving average")

        st.markdown("---")
        _rga, _rgb = st.columns([3, 1])

        _rmap = {"Bull Market":3,"Quiet Bull":2,"Transition":1,
                 "Quiet Bear":-1,"Bear Market":-2,"High Volatility":0}
        with _rga:
            st.markdown("**Regime Timeline**")
            fig_reg = go.Figure()
            fig_reg.add_trace(go.Scatter(
                x=_reg_df["Date"], y=_reg_df["Price_Gold"],
                line=dict(color="rgba(214,228,247,0.25)", width=1),
                name="Price", yaxis="y2"))
            for _r, _c in _reg_colors.items():
                _m = _reg_df["regime"] == _r
                if _m.any():
                    fig_reg.add_trace(go.Scatter(
                        x=_reg_df.loc[_m,"Date"],
                        y=[_rmap.get(_r,0)] * _m.sum(),
                        mode="markers",
                        marker=dict(color=_c, size=3, opacity=0.8),
                        name=_r, yaxis="y"))
            fig_reg.update_layout(
                height=280, template="plotly_dark",
                paper_bgcolor="#0d1b2a", plot_bgcolor="#061422",
                yaxis=dict(title="Regime", tickvals=list(_rmap.values()),
                           ticktext=list(_rmap.keys()), showgrid=False),
                yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Price (USD)"),
                hovermode="x unified",
                legend=dict(orientation="h", y=1.02, font=dict(size=9)),
                margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_reg, width="stretch")

        with _rgb:
            st.markdown("**Time in Each Regime**")
            fig_rd = go.Figure(go.Bar(
                x=list(_rdist.values), y=list(_rdist.index),
                orientation="h",
                marker_color=[_reg_colors.get(r,"#94a3b8") for r in _rdist.index],
                text=[f"{v:.0f}%" for v in _rdist.values],
                textposition="outside"))
            fig_rd.update_layout(
                height=280, template="plotly_dark",
                paper_bgcolor="#0d1b2a", plot_bgcolor="#061422",
                xaxis_title="% of time",
                margin=dict(l=0, r=30, t=10, b=0), showlegend=False)
            st.plotly_chart(fig_rd, width="stretch")





    st.markdown("---")
    with st.expander("View Raw Data"):
        _raw = disp_df[['Date', 'Price_Gold', 'High_Gold', 'Low_Gold',
                            'Price_Oil', 'Price_Dollar', 'Price_Stocks']].dropna()
        st.dataframe(_raw.set_index('Date'), width='stretch')
        st.download_button(
            label="Download as CSV",
            data=_raw.to_csv(index=False).encode('utf-8'),
            file_name=f"gold_data_{start_date}_{end_date}_{_ulbl}.csv",
            mime="text/csv",
            use_container_width=True)



# ══════════════════════════════════════════════════════════════════════════════
# SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Simulator":

    page_header("Investment Simulator",
                "Compare gold vs other assets over any historical period — with CAGR, drawdown & Sharpe ratio.")

    # Date range controls (inline since sidebar hides it for this page)
    sd1, sd2 = st.columns(2)
    with sd1:
        sim_start = st.date_input("From", value=pd.to_datetime("2010-01-01").date(),
                                  min_value=df["Date"].min().date(),
                                  max_value=df["Date"].max().date(), key="sim_from")
    with sd2:
        sim_end = st.date_input("To", value=df["Date"].max().date(),
                                min_value=df["Date"].min().date(),
                                max_value=df["Date"].max().date(), key="sim_to")

    sim_mask   = (df["Date"].dt.date >= sim_start) & (df["Date"].dt.date <= sim_end)
    sim_base   = df[sim_mask].copy()

    st.markdown("---")
    ctrl1, ctrl2 = st.columns([1, 3])
    with ctrl1:
        invest_amount = st.number_input("Initial Investment (USD)",
                                        min_value=100, max_value=10_000_000,
                                        value=10_000, step=1_000, key="sim_amount")
        sim_assets = st.multiselect("Compare with",
                                    ["Oil", "S&P 500", "Dollar Index"],
                                    default=["S&P 500"], key="sim_assets")

    asset_col_map   = {"Gold": "Price_Gold", "Oil": "Price_Oil",
                       "S&P 500": "Price_Stocks", "Dollar Index": "Price_Dollar"}
    selected_assets = ["Gold"] + sim_assets
    sim_cols = ["Date"] + [asset_col_map[a] for a in selected_assets
                           if asset_col_map[a] in sim_base.columns]
    sim_df   = sim_base[sim_cols].dropna().sort_values("Date").reset_index(drop=True)

    if sim_df.empty:
        st.warning("No data in selected range.")
        st.stop()

    sim_years = (sim_df["Date"].iloc[-1] - sim_df["Date"].iloc[0]).days / 365.25
    results   = {}
    for asset in selected_assets:
        col = asset_col_map.get(asset)
        if col and col in sim_df.columns:
            p0 = sim_df[col].iloc[0]
            p1 = sim_df[col].iloc[-1]
            if p0 > 0:
                final_val = invest_amount * (p1 / p0)
                total_ret = (p1 / p0 - 1) * 100
                cagr      = ((p1 / p0) ** (1 / sim_years) - 1) * 100 if sim_years > 0 else 0
                results[asset] = {"final_val": final_val, "total_ret": total_ret, "cagr": cagr}

    best_asset = max(results, key=lambda a: results[a]["final_val"]) if results else None

    # KPI row
    kpi_cols = st.columns(len(results))
    for i, (asset, res) in enumerate(results.items()):
        label = f"{asset} (Best)" if asset == best_asset else asset
        kpi_cols[i].metric(label, f"${res['final_val']:,.0f}",
                           f"{res['total_ret']:+.1f}%  |  CAGR {res['cagr']:+.1f}%")

    st.markdown("---")

    # Portfolio growth chart
    fig_sim  = go.Figure()
    pal_sim  = {"Gold": "#f2ca50", "Oil": "#2e5f65",
                "S&P 500": "#e0f7fa", "Dollar Index": "#4a5568"}
    for asset in selected_assets:
        col = asset_col_map.get(asset)
        if col and col in sim_df.columns:
            p0     = sim_df[col].iloc[0]
            series = (sim_df[col] / p0) * invest_amount
            fig_sim.add_trace(go.Scatter(
                x=sim_df["Date"], y=series, mode="lines", name=asset,
                line=dict(color=pal_sim.get(asset, "#ffffff"), width=2),
                hovertemplate="%{x|%b %d, %Y}<br>$%{y:,.0f}<extra>" + asset + "</extra>"))
    fig_sim.add_hline(y=invest_amount,
                      line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dot"),
                      annotation_text=f"Initial ${invest_amount:,.0f}",
                      annotation_position="bottom right",
                      annotation=dict(font_color="rgba(255,255,255,0.4)", font_size=10))
    fig_sim.update_layout(height=400, template="plotly_dark",
                          paper_bgcolor="#1c1f23", plot_bgcolor="#09090b",
                          yaxis_title="Portfolio Value (USD)", hovermode="x unified",
                          legend=dict(orientation="h", y=1.02, font=dict(size=11)),
                          margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_sim, width="stretch")

    # Drawdown chart
    st.markdown("**Gold Drawdown from Peak**")
    gold_series = (sim_df["Price_Gold"] / sim_df["Price_Gold"].iloc[0]) * invest_amount
    rolling_max = gold_series.cummax()
    drawdown    = (gold_series - rolling_max) / rolling_max * 100
    max_dd      = drawdown.min()
    worst_date  = sim_df["Date"].iloc[drawdown.idxmin()].strftime("%b %Y")

    dd1, dd2 = st.columns(2)
    dd1.metric("Max Drawdown", f"{max_dd:.1f}%", worst_date)
    dd2.metric("Period", f"{sim_years:.1f} years",
               f"{(sim_df['Date'].iloc[-1] - sim_df['Date'].iloc[0]).days:,} days")

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=sim_df["Date"], y=drawdown,
                                 fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
                                 line=dict(color="#ef4444", width=1.2),
                                 name="Gold Drawdown",
                                 hovertemplate="%{x|%b %d, %Y}<br>%{y:.1f}%<extra></extra>"))
    fig_dd.update_layout(height=200, template="plotly_dark",
                         paper_bgcolor="#1c1f23", plot_bgcolor="#09090b",
                         yaxis=dict(title="Drawdown (%)", ticksuffix="%"),
                         margin=dict(l=0, r=0, t=4, b=0), showlegend=False)
    st.plotly_chart(fig_dd, width="stretch")

    # Summary table
    st.markdown("---")
    if results:
        _tbl = pd.DataFrame([
            {"Asset": a, "Start Value": f"${invest_amount:,.0f}",
             "End Value": f"${r['final_val']:,.0f}",
             "Total Return": f"{r['total_ret']:+.1f}%",
             "CAGR": f"{r['cagr']:+.1f}%"}
            for a, r in results.items()
        ]).set_index("Asset")
        st.dataframe(_tbl, width="stretch")
        st.download_button(
            label="Download Results as CSV",
            data=_tbl.reset_index().to_csv(index=False).encode("utf-8"),
            file_name=f"simulation_{sim_start}_{sim_end}.csv",
            mime="text/csv",
            use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Prediction":

    page_header("Gold Price Prediction",
                "Estimates short-term market tendencies using historical & macro indicators. Financial markets are partially stochastic — no model predicts prices with certainty.")

    st.markdown(
        '<div style="background:rgba(242,202,80,0.05);border:1px solid rgba(242,202,80,0.2);'
        'border-left:3px solid #f2ca50;border-radius:8px;padding:8px 14px;'
        'font-size:0.78rem;color:#d6e4f7;opacity:0.8;margin-bottom:16px;">'
        '<b style="color:#f2ca50;">Research Disclaimer:</b> This tool estimates short-term market '
        'tendencies using historical & macro-financial indicators. Financial markets contain '
        'significant stochastic components — outputs should not be used as investment advice.'
        '</div>',
        unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        model_name = st.selectbox("Model",
                                   ["Random Forest", "XGBoost", "LightGBM", "Linear Regression", "Neural Network (MLP)"],
                                   help="Choose the ML algorithm. LightGBM & XGBoost are generally best for financial time series.")
        if model_name == "Neural Network (MLP)":
            st.warning("MLP without sequence input rarely outperforms tree models on tabular financial data. "
                       "Consider XGBoost or LightGBM for better results.")
    with c2:
        test_pct = st.slider("Test Size %", 10, 40, 20,
                           help="What % of data to use for testing. Walk-Forward splits this into multiple folds.")
    with c3:
        n_lags = st.slider("Lag Features (days)", 1, 30, 5,
                           help="How many past days of returns to use as input. More lags = more memory but more noise.")

    _tech_opts  = [c for c in ['EMA10_pct','EMA20_pct','MACD_pct','ATR_pct'] if c in df.columns]
    _eco_opts   = [c for c in ['VIX','TNX_Yield'] if c in df.columns]
    extra_feats = st.multiselect(
        "Additional Features",
        ['Price_Oil','Price_Dollar','Price_Stocks'] + _eco_opts + _tech_opts,
        default=['Price_Oil','Price_Dollar','MACD_pct','ATR_pct'] +
                [c for c in ['VIX','TNX_Yield'] if c in df.columns])

    run = st.button("Train Model", type="primary", width='stretch')

    if run:
        with st.spinner("Training ... please wait"):

            cols = ['Date', 'Price_Gold'] + extra_feats
            ml   = filtered_df[cols].dropna().sort_values('Date').reset_index(drop=True)

            # ── تحويل للعوائد اليومية (stationary) ───────────────────────────
            ml['Return_Gold'] = ml['Price_Gold'].pct_change() * 100   # % يومي

            # ── 1. Lag Features (return lags — stationary) ───────────────
            # slider lags (1..n_lags)
            for lag in range(1, n_lags + 1):
                ml[f'lag_{lag}'] = ml['Return_Gold'].shift(lag)
            # Fixed meaningful lags regardless of slider
            for lag in [7, 14, 21]:
                if lag > n_lags:  # avoid duplication
                    ml[f'lag_{lag}'] = ml['Return_Gold'].shift(lag)

            # ── 2. Rolling Statistics (mean & std of returns) ────────────
            for _w in [5, 10, 14, 20, 30]:
                ml[f'rolling_mean_{_w}'] = ml['Return_Gold'].rolling(_w).mean()
                ml[f'rolling_std_{_w}']  = ml['Return_Gold'].rolling(_w).std()

            # ── 3. Volatility Features ────────────────────────────────────
            # Realized volatility: annualised std of returns over window
            ml['vol_5']  = ml['Return_Gold'].rolling(5).std()  * np.sqrt(252)
            ml['vol_21'] = ml['Return_Gold'].rolling(21).std() * np.sqrt(252)
            ml['vol_63'] = ml['Return_Gold'].rolling(63).std() * np.sqrt(252)
            # Volatility ratio: short-term vs long-term (regime signal)
            ml['vol_ratio'] = ml['vol_5'] / (ml['vol_63'] + 1e-9)

            # ── 4. Momentum Features ──────────────────────────────────────
            # Price momentum: % change over N days
            for _d in [5, 7, 10, 14, 21, 30]:
                ml[f'mom_{_d}d'] = ml['Price_Gold'].pct_change(_d) * 100
            # Rate of Change (ROC)
            ml['roc_5']  = (ml['Price_Gold'] - ml['Price_Gold'].shift(5))  / (ml['Price_Gold'].shift(5)  + 1e-9) * 100
            ml['roc_20'] = (ml['Price_Gold'] - ml['Price_Gold'].shift(20)) / (ml['Price_Gold'].shift(20) + 1e-9) * 100
            # Mean-reversion: distance from 20-day SMA (z-score)
            _sma20 = ml['Price_Gold'].rolling(20).mean()
            _std20 = ml['Price_Gold'].rolling(20).std()
            ml['zscore_20'] = (ml['Price_Gold'] - _sma20) / (_std20 + 1e-9)

            # ── 5. Price assets -> returns; indicator features used directly
            price_feats = [f for f in extra_feats if f.startswith('Price_')]
            indic_feats = [f for f in extra_feats if not f.startswith('Price_')]
            # VIX and TNX_Yield are level features — normalise as % change
            for _ef in ['VIX','TNX_Yield']:
                if _ef in indic_feats and _ef in ml.columns:
                    ml[f'{_ef}_ret'] = ml[_ef].pct_change() * 100
                    indic_feats = [f'{_ef}_ret' if x == _ef else x for x in indic_feats]
            for feat in price_feats:
                ml[f'{feat}_ret'] = ml[feat].pct_change() * 100

            ml = ml.dropna().reset_index(drop=True)

            ret_extra    = [f'{f}_ret' for f in price_feats if f'{f}_ret' in ml.columns] + \
                           [f for f in indic_feats if f in ml.columns]
            _fixed_lags  = [f'lag_{l}' for l in [7,14,21] if l > n_lags]
            _roll_feats  = [f'rolling_mean_{w}' for w in [5,10,14,20,30]] + \
                           [f'rolling_std_{w}'  for w in [5,10,14,20,30]]
            _vol_feats   = ['vol_5','vol_21','vol_63','vol_ratio']
            _mom_feats   = [f'mom_{d}d' for d in [5,7,10,14,21,30]] + \
                           ['roc_5','roc_20','zscore_20']
            feature_cols = ([f'lag_{i}' for i in range(1, n_lags + 1)] +
                            _fixed_lags + ret_extra +
                            _roll_feats + _vol_feats + _mom_feats)
            feature_cols = [c for c in feature_cols if c in ml.columns]

            X      = ml[feature_cols].values
            y_ret  = ml['Return_Gold'].values      # هدف: العائد اليومي %
            prices = ml['Price_Gold'].values        # للعرض فقط
            dates  = ml['Date'].values

            # ── Build model factory ───────────────────────────────────────
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.preprocessing import StandardScaler

            def _make_model():
                if model_name == "Random Forest":
                    return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
                elif model_name == "Linear Regression":
                    return LinearRegression()
                elif model_name == "LightGBM":
                    try:
                        from lightgbm import LGBMRegressor
                        return LGBMRegressor(n_estimators=500, learning_rate=0.03,
                                             num_leaves=63, min_child_samples=20,
                                             subsample=0.8, colsample_bytree=0.8,
                                             random_state=42, verbose=-1)
                    except ImportError:
                        st.error("LightGBM not available."); st.stop()
                elif model_name == "XGBoost":
                    try:
                        from xgboost import XGBRegressor
                        return XGBRegressor(n_estimators=300, learning_rate=0.05,
                                            max_depth=6, random_state=42, verbosity=0)
                    except ImportError:
                        st.error("XGBoost not available."); st.stop()
                else:  # MLP
                    from sklearn.neural_network import MLPRegressor
                    return MLPRegressor(hidden_layer_sizes=(128, 64, 32),
                                        activation="relu", solver="adam",
                                        max_iter=500, random_state=42,
                                        early_stopping=True, n_iter_no_change=20)

            # ── Walk-Forward Validation (TimeSeriesSplit) ─────────────────
            n_splits  = min(5, int(len(X) * (1 - test_pct/100) / max(int(len(X)*test_pct/100/5), 1)))
            n_splits  = max(3, min(n_splits, 5))
            tscv      = TimeSeriesSplit(n_splits=n_splits)

            fold_metrics  = []   # per-fold results
            all_dates_te  = []
            all_prices_te = []
            all_preds_ret = []
            all_preds_pr  = []
            all_y_te      = []

            for fold_i, (tr_idx, te_idx) in enumerate(tscv.split(X)):
                X_tr, X_te = X[tr_idx], X[te_idx]
                y_tr, y_te_f = y_ret[tr_idx], y_ret[te_idx]
                p_te   = prices[te_idx]
                d_te   = dates[te_idx]

                _mdl = _make_model()
                if model_name == "Neural Network (MLP)":
                    _sx = StandardScaler(); _sy = StandardScaler()
                    _mdl.fit(_sx.fit_transform(X_tr),
                             _sy.fit_transform(y_tr.reshape(-1,1)).ravel())
                    _pr = _sy.inverse_transform(
                        _mdl.predict(_sx.transform(X_te)).reshape(-1,1)).ravel()
                else:
                    _mdl.fit(X_tr, y_tr)
                    _pr = _mdl.predict(X_te)

                # 1-step-ahead price reconstruction
                _pp = np.zeros(len(p_te))
                _pp[0] = p_te[0]
                for _i in range(1, len(p_te)):
                    _pp[_i] = p_te[_i-1] * (1 + _pr[_i] / 100)

                _da = float(np.mean(np.sign(y_te_f[1:]) == np.sign(_pr[1:])) * 100)
                _rm = float(np.sqrt(mean_squared_error(p_te[1:], _pp[1:])))
                _ma = float(mean_absolute_error(p_te[1:], _pp[1:]))
                _r2 = float(r2_score(y_te_f, _pr))
                fold_metrics.append({"Fold": fold_i+1, "Dir Acc %": round(_da,1),
                                     "RMSE $": round(_rm,2), "MAE $": round(_ma,2),
                                     "R² (ret)": round(_r2,4),
                                     "Test days": len(te_idx)})
                all_dates_te.append(d_te)
                all_prices_te.append(p_te)
                all_preds_ret.append(_pr)
                all_preds_pr.append(_pp)
                all_y_te.append(y_te_f)

            # Use last fold as primary result + keep last model for signal
            mdl       = _mdl
            prices_te = all_prices_te[-1]
            dates_te  = all_dates_te[-1]
            preds_ret = all_preds_ret[-1]
            preds_prices = all_preds_pr[-1]
            y_te      = all_y_te[-1]

            # Aggregate metrics across all folds
            avg_da  = float(np.mean([f["Dir Acc %"] for f in fold_metrics]))
            avg_rm  = float(np.mean([f["RMSE $"]    for f in fold_metrics]))
            avg_ma  = float(np.mean([f["MAE $"]     for f in fold_metrics]))
            avg_r2  = float(np.mean([f["R² (ret)"]  for f in fold_metrics]))
            rmse_price = avg_rm
            mae_price  = avg_ma

            # Concatenated predictions for full-history chart
            all_dates_cat  = np.concatenate(all_dates_te)
            all_prices_cat = np.concatenate(all_prices_te)
            all_preds_cat  = np.concatenate(all_preds_pr)
            all_y_cat      = np.concatenate(all_y_te)
            all_predr_cat  = np.concatenate(all_preds_ret)

            # ── Metrics ──────────────────────────────────────────────────────
            # ── Walk-Forward aggregated metrics ────────────────────────────
            rmse_ret   = float(np.sqrt(mean_squared_error(all_y_cat, all_predr_cat)))
            mae_ret    = float(mean_absolute_error(all_y_cat, all_predr_cat))
            r2_ret     = float(r2_score(all_y_cat, all_predr_cat))
            rmse_price = avg_rm
            mae_price  = avg_ma
            dir_acc    = avg_da

            st.success(f"{model_name} trained — Walk-Forward Validation across {n_splits} folds.")

            st.markdown(f"""
            <div style="background:#13212e;border:1px solid rgba(255,255,255,0.08);border-radius:8px;
                        padding:10px 16px;margin-bottom:12px;font-size:0.82rem;color:#d6e4f7;opacity:0.8;">
            <b>Walk-Forward Validation</b> — {n_splits} folds, always training on the past and testing on the future.
            <b>1-step-ahead:</b> each day anchored to yesterday's actual price. No error compounding.
            All metrics are averages across folds.
            </div>
            """, unsafe_allow_html=True)

            mc1, mc2, mc3 = st.columns(3)
            dir_delta = "above random" if dir_acc >= 52 else ("at random" if dir_acc >= 48 else "below random")
            mc1.metric("Directional Accuracy", f"{dir_acc:.1f}%", dir_delta)
            mc2.metric("RMSE (price $)",       f"${rmse_price:,.2f}")
            mc3.metric("MAE (price $)",        f"${mae_price:,.2f}")

            mc4, mc5, mc6 = st.columns(3)
            mc4.metric("R² (returns)",    f"{r2_ret:.4f}")
            mc5.metric("RMSE (return %)", f"{rmse_ret:.4f}%")
            mc6.metric("MAE (return %)",  f"{mae_ret:.4f}%")

            with st.expander(f"Walk-Forward Fold Breakdown ({n_splits} folds)"):
                _fdf = pd.DataFrame(fold_metrics).set_index("Fold")
                st.dataframe(_fdf, width="stretch")
                _fold_fig = go.Figure()
                _fold_fig.add_trace(go.Bar(x=_fdf.index, y=_fdf["Dir Acc %"],
                                           marker_color=["#22c55e" if v>=52 else "#ef4444"
                                                         for v in _fdf["Dir Acc %"]],
                                           name="Dir Accuracy %"))
                _fold_fig.add_hline(y=50, line=dict(color="#f2ca50", width=1, dash="dot"),
                                    annotation_text="Random baseline 50%")
                _fold_fig.update_layout(height=220, template="plotly_dark",
                                        paper_bgcolor="#0d1b2a", plot_bgcolor="#061422",
                                        xaxis_title="Fold", yaxis_title="Dir Accuracy %",
                                        margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
                st.plotly_chart(_fold_fig, width="stretch")

            # ── مخطط الأسعار الفعلية vs المُعادة ─────────────────────────────

            # ── Model Comparison ─────────────────────────────────────────
            st.markdown('---')
            with st.expander("Compare All Models", expanded=False):
                _all_models = {
                    "Random Forest":        lambda: RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
                    "XGBoost":              None,
                    "LightGBM":             None,
                    "Linear Regression":    lambda: LinearRegression(),
                }
                _cmp_rows = []
                _cmp_fig  = go.Figure()
                _pal_cmp  = {"Random Forest":"#f2ca50","XGBoost":"#2e5f65",
                             "LightGBM":"#22c55e","Linear Regression":"#e0f7fa"}
                with st.spinner("Running all models for comparison..."):
                    for _mn, _factory in _all_models.items():
                        try:
                            if _mn == "XGBoost":
                                from xgboost import XGBRegressor
                                _m = XGBRegressor(n_estimators=300, learning_rate=0.05,
                                                  max_depth=6, random_state=42, verbosity=0)
                            elif _mn == "LightGBM":
                                from lightgbm import LGBMRegressor
                                _m = LGBMRegressor(n_estimators=500, learning_rate=0.03,
                                                   num_leaves=63, min_child_samples=20,
                                                   subsample=0.8, colsample_bytree=0.8,
                                                   random_state=42, verbose=-1)
                            else:
                                _m = _factory()
                            _m.fit(X_tr, y_tr)
                            _pr = _m.predict(X_te)
                            _pp = np.zeros(len(prices_te))
                            _pp[0] = prices_te[0]
                            for _i in range(1, len(prices_te)):
                                _pp[_i] = prices_te[_i-1] * (1 + _pr[_i] / 100)
                            _da  = float(np.mean(np.sign(y_te[1:]) == np.sign(_pr[1:])) * 100)
                            _rm  = float(np.sqrt(mean_squared_error(prices_te[1:], _pp[1:])))
                            _ma  = float(mean_absolute_error(prices_te[1:], _pp[1:]))
                            _r2  = float(r2_score(y_te, _pr))
                            _cmp_rows.append({"Model": _mn,
                                              "Dir. Acc %": f"{_da:.1f}",
                                              "RMSE $": f"{_rm:,.2f}",
                                              "MAE $": f"{_ma:,.2f}",
                                              "R² (returns)": f"{_r2:.4f}"})
                            _cum = np.cumprod(1 + np.where(_pr[:-1]>0, y_te[1:], 0.0) / 100)
                            _cmp_fig.add_trace(go.Scatter(
                                x=dates_te[1:], y=_cum, name=_mn,
                                line=dict(color=_pal_cmp.get(_mn,"#ffffff"), width=1.8)))
                        except Exception:
                            pass
                if _cmp_rows:
                    _cmp_df = pd.DataFrame(_cmp_rows).set_index("Model")
                    # highlight best values
                    st.dataframe(_cmp_df, width='stretch')
                    _cmp_fig.add_trace(go.Scatter(
                        x=dates_te[1:],
                        y=np.cumprod(1 + y_te[1:] / 100),
                        name="Buy & Hold",
                        line=dict(color='#4a5568', width=1.5, dash='dot')))
                    _cmp_fig.update_layout(
                        height=260, template='plotly_dark',
                        paper_bgcolor='#0d1b2a', plot_bgcolor='#061422',
                        yaxis_title="Growth ($1)", hovermode='x unified',
                        title="All Models — Cumulative Strategy Return",
                        legend=dict(orientation='h', y=1.08, font=dict(size=10)),
                        margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(_cmp_fig, width='stretch')

            # ══════════════════════════════════════════════════════════════
            # BACKTESTING ENGINE (Walk-Forward out-of-sample only)
            # ══════════════════════════════════════════════════════════════
            st.markdown("---")
            st.subheader("Backtesting")
            st.markdown(
                '<div style="background:#13212e;border:1px solid rgba(255,255,255,0.08);border-radius:8px;'
                'padding:10px 16px;margin-bottom:14px;font-size:0.82rem;color:#d6e4f7;opacity:0.8;">'
                '<b>Honest backtesting:</b> uses only Walk-Forward out-of-sample predictions — '
                'the model never saw the test data during training. '
                'Strategy: <b style="color:#f2ca50;">Long when predicted return &gt; 0.05%</b>, otherwise Cash.</div>',
                unsafe_allow_html=True)

            # ── Quick summary table (most important for quant reviewers) ────
            _bt_preview_data = {
                "Metric":   ["Total Return", "Annual Return", "Max Drawdown",
                             "Sharpe Ratio", "Sortino Ratio", "Win Rate", "Calmar Ratio"],
                "Strategy": ["—"]*7,
                "Buy & Hold": ["—"]*7,
                "Better?": ["—"]*7,
            }
            _bt_placeholder = st.empty()


            _bt_pred   = all_predr_cat
            _bt_actual = all_y_cat
            _bt_dates  = all_dates_cat

            _position = (_bt_pred > 0.05).astype(float)
            _strat_r  = _position * _bt_actual
            _bh_r     = _bt_actual
            _cum_s    = np.cumprod(1 + _strat_r  / 100)
            _cum_bh   = np.cumprod(1 + _bh_r     / 100)

            def _bt_metrics(rets, cum):
                n       = max(len(rets), 1)
                tot_ret = float((cum[-1] - 1) * 100)
                ann_ret = float((cum[-1] ** (252/n) - 1) * 100)
                peak    = np.maximum.accumulate(cum)
                dd      = (cum - peak) / peak * 100
                max_dd  = float(dd.min())
                mu, sig = np.mean(rets), np.std(rets)
                sharpe  = float(mu / sig * np.sqrt(252)) if sig > 1e-9 else 0.0
                down    = np.std(rets[rets < 0]) if np.any(rets < 0) else 1e-9
                sortino = float(mu / down * np.sqrt(252)) if down > 1e-9 else 0.0
                calmar  = float(ann_ret / abs(max_dd)) if abs(max_dd) > 1e-9 else 0.0
                return dict(tot=tot_ret, ann=ann_ret, dd=max_dd,
                            sharpe=sharpe, sortino=sortino, calmar=calmar)

            _ms = _bt_metrics(_strat_r, _cum_s)
            _bh = _bt_metrics(_bh_r,    _cum_bh)

            _trade_mask = _position > 0
            _trade_rets = _bt_actual[_trade_mask]
            _win_rate   = float(np.mean(_trade_rets > 0) * 100) if len(_trade_rets) > 0 else 0.0
            _n_trades   = int(np.sum(np.diff(np.concatenate([[0], _trade_mask.astype(int)])) == 1))
            _avg_win    = float(np.mean(_trade_rets[_trade_rets > 0])) if np.any(_trade_rets > 0) else 0.0
            _avg_loss   = float(np.mean(_trade_rets[_trade_rets < 0])) if np.any(_trade_rets < 0) else 0.0
            _coverage   = int(np.mean(_position) * 100)

            def _mkpi(label, sv, bv, fmt, good_high=True):
                delta = sv - bv
                color = "#22c55e" if (delta > 0) == good_high else "#ef4444"
                return (
                    f'<div style="background:#13212e;border:1px solid rgba(255,255,255,0.08);'

                    f'border-top:2px solid #f2ca50;border-radius:12px;padding:14px 16px;">'

                    f'<div style="font-size:.7rem;color:#d6e4f7;opacity:.6;text-transform:uppercase;'

                    f'letter-spacing:.05em;">{label}</div>'

                    f'<div style="font-family:monospace;font-size:1.35rem;font-weight:700;color:#f2ca50;">{fmt.format(sv)}</div>'

                    f'<div style="font-size:.72rem;color:{color};">vs B&H {fmt.format(bv)}&nbsp;({delta:+.2f})</div>'

                    f'</div>')

            _c1,_c2,_c3,_c4,_c5,_c6 = st.columns(6)
            for _col, _l, _s, _b, _f, _g in [
                (_c1,"Total Return",   _ms["tot"],    _bh["tot"],    "{:.1f}%", True),
                (_c2,"Annual Return",  _ms["ann"],    _bh["ann"],    "{:.1f}%", True),
                (_c3,"Max Drawdown",   _ms["dd"],     _bh["dd"],     "{:.1f}%", False),
                (_c4,"Sharpe Ratio",   _ms["sharpe"], _bh["sharpe"], "{:.2f}",  True),
                (_c5,"Sortino Ratio",  _ms["sortino"],_bh["sortino"],"{:.2f}",  True),
                (_c6,"Win Rate",       _win_rate,     50.0,          "{:.1f}%", True),
            ]:
                _col.markdown(_mkpi(_l,_s,_b,_f,_g), unsafe_allow_html=True)

            # ── Quant-style performance summary table ─────────────────────────
            _tbl_rows = [
                ("Total Return",   f"{_ms['tot']:.1f}%",     f"{_bh['tot']:.1f}%",     _ms['tot']    >  _bh['tot']),
                ("Annual Return",  f"{_ms['ann']:.1f}%",     f"{_bh['ann']:.1f}%",     _ms['ann']    >  _bh['ann']),
                ("Max Drawdown",   f"{_ms['dd']:.1f}%",      f"{_bh['dd']:.1f}%",      _ms['dd']     >  _bh['dd']),
                ("Sharpe Ratio",   f"{_ms['sharpe']:.2f}",   f"{_bh['sharpe']:.2f}",   _ms['sharpe'] >  _bh['sharpe']),
                ("Sortino Ratio",  f"{_ms['sortino']:.2f}",  f"{_bh['sortino']:.2f}",  _ms['sortino']>  _bh['sortino']),
                ("Win Rate",       f"{_win_rate:.1f}%",       "50.0%",                  _win_rate     >  50),
                ("Calmar Ratio",   f"{_ms['calmar']:.2f}",   f"{_bh['calmar']:.2f}",   _ms['calmar'] >  _bh['calmar']),
            ]
            _th = "padding:8px 12px;font-size:.75rem;text-transform:uppercase;letter-spacing:.05em;"
            _td = "padding:7px 12px;font-family:monospace;font-size:.82rem;"
            _tbl = (
                '<table style="width:100%;border-collapse:collapse;">'
                f'<thead><tr>'
                f'<th style="{_th}text-align:left;color:#d6e4f7;opacity:.6;border-bottom:1px solid rgba(255,255,255,0.1);">Metric</th>'
                f'<th style="{_th}text-align:right;color:#f2ca50;border-bottom:1px solid rgba(255,255,255,0.1);">Model Strategy</th>'
                f'<th style="{_th}text-align:right;color:#d6e4f7;opacity:.7;border-bottom:1px solid rgba(255,255,255,0.1);">Buy & Hold</th>'
                f'<th style="{_th}text-align:center;color:#d6e4f7;opacity:.6;border-bottom:1px solid rgba(255,255,255,0.1);">Edge?</th>'
                f'</tr></thead><tbody>'
            )
            for _metric, _sv, _bv, _wins in _tbl_rows:
                _is_dd = "Drawdown" in _metric
                _edge  = not _wins if _is_dd else _wins
                _ec    = "#22c55e" if _edge else "#ef4444"
                _icon  = "✓ Yes" if _edge else "✗ No"
                _tbl  += (
                    f'<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">'
                    f'<td style="{_td}color:#d6e4f7;">{_metric}</td>'
                    f'<td style="{_td}text-align:right;font-weight:700;color:#f2ca50;">{_sv}</td>'
                    f'<td style="{_td}text-align:right;color:#d6e4f7;opacity:.7;">{_bv}</td>'
                    f'<td style="{_td}text-align:center;color:{_ec};font-weight:700;">{_icon}</td>'
                    f'</tr>'
                )
            _tbl += '</tbody></table>'
            st.markdown(
                '<div style="background:#13212e;border:1px solid rgba(255,255,255,0.08);'
                'border-radius:12px;padding:18px;margin-bottom:16px;">'
                '<div style="font-size:.7rem;color:#d6e4f7;opacity:.6;text-transform:uppercase;'
                'letter-spacing:.05em;margin-bottom:12px;">Performance Summary vs Buy & Hold (Out-of-Sample)</div>'
                + _tbl + '</div>',
                unsafe_allow_html=True)

            st.markdown(
                f'<div style="background:#0d1b2a;border:1px solid rgba(255,255,255,0.06);'
                f'border-radius:8px;padding:8px 16px;margin:10px 0;font-size:0.8rem;color:#d6e4f7;opacity:.7;">'
                f'Trades: <b style="color:#f2ca50">{_n_trades}</b> &nbsp;·&nbsp;'
                f'Avg Win: <b style="color:#22c55e">+{_avg_win:.3f}%</b> &nbsp;·&nbsp;'
                f'Avg Loss: <b style="color:#ef4444">{_avg_loss:.3f}%</b> &nbsp;·&nbsp;'
                f'Calmar: <b style="color:#f2ca50">{_ms["calmar"]:.2f}</b> &nbsp;·&nbsp;'
                f'Days in Market: <b>{_coverage}%</b></div>',
                unsafe_allow_html=True)


            st.markdown("---")

            _eq_fig = go.Figure()
            _eq_fig.add_trace(go.Scatter(x=_bt_dates, y=_cum_bh*1000,
                name="Buy & Hold", line=dict(color="#d6e4f7", width=1.8)))
            _eq_fig.add_trace(go.Scatter(x=_bt_dates, y=_cum_s*1000,
                name=f"Model Strategy ({model_name})",
                line=dict(color="#f2ca50", width=2.2)))
            _eq_fig.update_layout(height=300, template="plotly_dark",
                paper_bgcolor="#0d1b2a", plot_bgcolor="#061422",
                title="Equity Curve — $1,000 initial investment",
                yaxis_title="Portfolio Value ($)", hovermode="x unified",
                legend=dict(orientation="h", y=1.02, font=dict(size=10)),
                margin=dict(l=0, r=0, t=36, b=0))
            st.plotly_chart(_eq_fig, width="stretch")

            _peak_s  = np.maximum.accumulate(_cum_s)
            _peak_bh = np.maximum.accumulate(_cum_bh)
            _dd_s    = (_cum_s  - _peak_s)  / _peak_s  * 100
            _dd_bh   = (_cum_bh - _peak_bh) / _peak_bh * 100
            _dd_fig  = go.Figure()
            _dd_fig.add_trace(go.Scatter(x=_bt_dates, y=_dd_bh,
                fill="tozeroy", fillcolor="rgba(239,68,68,0.08)",
                line=dict(color="rgba(239,68,68,0.4)", width=1), name="B&H Drawdown"))
            _dd_fig.add_trace(go.Scatter(x=_bt_dates, y=_dd_s,
                fill="tozeroy", fillcolor="rgba(242,202,80,0.08)",
                line=dict(color="#f2ca50", width=1.5), name="Strategy Drawdown"))
            _dd_fig.update_layout(height=200, template="plotly_dark",
                paper_bgcolor="#0d1b2a", plot_bgcolor="#061422",
                yaxis=dict(title="Drawdown (%)", ticksuffix="%"),
                hovermode="x unified",
                legend=dict(orientation="h", y=1.02, font=dict(size=10)),
                margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(_dd_fig, width="stretch")

            st.markdown("---")
            last_pred_ret = float(preds_ret[-1])
            if last_pred_ret > 0.15:
                sig_txt, sig_color = "Buy", "#22c55e"
                sig_desc = f"Model expects +{last_pred_ret:.2f}% next session"
            elif last_pred_ret < -0.15:
                sig_txt, sig_color = "Sell / Avoid", "#ef4444"
                sig_desc = f"Model expects {last_pred_ret:.2f}% next session"
            else:
                sig_txt, sig_color = "Hold / Neutral", "#f2ca50"
                sig_desc = f"Weak signal ({last_pred_ret:+.2f}%)"
            st.markdown(
                f'<div style="background:#13212e;border:1px solid rgba(255,255,255,0.08);'

                f'border-radius:12px;padding:20px;text-align:center;max-width:340px;margin:0 auto;">'

                f'<div style="font-size:.75rem;color:#d6e4f7;opacity:.6;text-transform:uppercase;'

                f'letter-spacing:.05em;margin-bottom:6px;">Signal — Next Session</div>'

                f'<div style="font-size:2.2rem;font-weight:900;color:{sig_color};">{sig_txt}</div>'

                f'<div style="font-size:.78rem;color:#d6e4f7;opacity:.5;margin-top:6px;">{sig_desc}</div></div>',
                unsafe_allow_html=True)
            st.markdown("---")

            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=all_dates_cat, y=all_prices_cat,
                                           name="Actual Price",
                                           line=dict(color="#d6e4f7", width=1.8)))
            fig_pred.add_trace(go.Scatter(x=all_dates_cat, y=all_preds_cat,
                                           name=f"Predicted ({model_name})",
                                           line=dict(color="#f2ca50", width=1.8, dash="dash")))
            # Mark fold boundaries
            for _fd in all_dates_te[:-1]:
                fig_pred.add_shape(type="line",
                                   x0=str(pd.Timestamp(_fd[0]).date()),
                                   x1=str(pd.Timestamp(_fd[0]).date()),
                                   y0=0, y1=1, xref="x", yref="paper",
                                   line=dict(color="rgba(242,202,80,0.2)", width=1, dash="dot"))
            fig_pred.update_layout(height=440, template="plotly_dark",
                                   paper_bgcolor="#0d1b2a", plot_bgcolor="#061422",
                                   title=f"{model_name} — Walk-Forward ({n_splits} folds)",
                                   yaxis_title="Price (USD)",
                                   hovermode="x unified",
                                   legend=dict(orientation="h", y=1.02),
                                   margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_pred, width="stretch")

            # ── مخطط العوائد اليومية ──────────────────────────────────────────
            with st.expander("Daily Returns: Actual vs Predicted"):
                fig_ret = go.Figure()
                fig_ret.add_trace(go.Scatter(x=dates_te, y=y_te, name='Actual Return',
                                              line=dict(color='#e0f7fa', width=1.2)))
                fig_ret.add_trace(go.Scatter(x=dates_te, y=preds_ret,
                                              name='Predicted Return',
                                              line=dict(color='#f2ca50', width=1.2, dash='dash')))
                fig_ret.add_hline(y=0, line=dict(color='#4a5568', width=1, dash='dot'))
                fig_ret.update_layout(height=280, template='plotly_dark',
                                      paper_bgcolor='#0d1b2a', plot_bgcolor='#061422',
                                      yaxis_title="Daily Return (%)",
                                      hovermode='x unified',
                                      legend=dict(orientation='h', y=1.02),
                                      margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_ret, width='stretch')

            if hasattr(mdl, "feature_importances_"):
                st.subheader("Feature Importance")
                fi = pd.Series(mdl.feature_importances_,
                               index=feature_cols).sort_values(ascending=False)

                # ── Group features into categories ────────────────────────
                def _categorize(name):
                    if name.startswith("lag_"):             return "Lag (Return)"
                    if name.startswith("rolling_mean"):     return "Rolling Mean"
                    if name.startswith("rolling_std"):      return "Rolling Std"
                    if name.startswith("vol_"):             return "Volatility"
                    if name.startswith("mom_") or name.startswith("roc_") or name == "zscore_20":
                        return "Momentum"
                    if "Oil" in name:                       return "Oil"
                    if "Dollar" in name or "DXY" in name:  return "Dollar/DXY"
                    if "Stock" in name or "SP" in name:    return "S&P 500"
                    if "VIX" in name:                      return "VIX"
                    if "TNX" in name or "Yield" in name:   return "Treasury Yield"
                    if name in ["EMA10_pct","EMA20_pct","MACD_pct","ATR_pct"]:
                        return "Technical (EMA/MACD/ATR)"
                    return "Other"

                fi_df = pd.DataFrame({"feature": fi.index, "importance": fi.values})
                fi_df["category"] = fi_df["feature"].apply(_categorize)
                fi_df["pct"] = fi_df["importance"] / fi_df["importance"].sum() * 100

                _cat_colors = {
                    "Lag (Return)":        "#f2ca50",
                    "Rolling Mean":        "#60a5fa",
                    "Rolling Std":         "#818cf8",
                    "Volatility":          "#ef4444",
                    "Momentum":            "#22c55e",
                    "Oil":                 "#fb923c",
                    "Dollar/DXY":          "#a78bfa",
                    "S&P 500":             "#34d399",
                    "VIX":                 "#f87171",
                    "Treasury Yield":      "#fbbf24",
                    "Technical (EMA/MACD/ATR)": "#2dd4bf",
                    "Other":               "#94a3b8",
                }

                fi_col1, fi_col2 = st.columns([3, 2])

                with fi_col1:
                    st.markdown("**Top 20 Features**")
                    top20 = fi_df.head(20).sort_values("importance")
                    _bar_colors = [_cat_colors.get(c,"#94a3b8") for c in top20["category"]]
                    fig_fi = go.Figure(go.Bar(
                        x=top20["pct"], y=top20["feature"],
                        orientation="h", marker_color=_bar_colors,
                        text=[f"{v:.1f}%" for v in top20["pct"]],
                        textposition="outside"))
                    fig_fi.update_layout(
                        height=480, template="plotly_dark",
                        paper_bgcolor="#0d1b2a", plot_bgcolor="#061422",
                        xaxis_title="Importance (%)",
                        margin=dict(l=0, r=40, t=10, b=0))
                    st.plotly_chart(fig_fi, width="stretch")

                with fi_col2:
                    st.markdown("**By Category**")
                    cat_sum = fi_df.groupby("category")["pct"].sum().sort_values(ascending=False)
                    fig_pie = go.Figure(go.Pie(
                        labels=cat_sum.index, values=cat_sum.values,
                        marker_colors=[_cat_colors.get(c,"#94a3b8") for c in cat_sum.index],
                        hole=0.45,
                        textinfo="label+percent",
                        textfont_size=11))
                    fig_pie.update_layout(
                        height=480, template="plotly_dark",
                        paper_bgcolor="#0d1b2a",
                        showlegend=False,
                        margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_pie, width="stretch")

                    # Top driver card
                    _top_cat = cat_sum.index[0]
                    _top_pct = cat_sum.values[0]
                    _top_feat = fi_df.iloc[0]["feature"]
                    st.markdown(
                        f'<div style="background:#13212e;border:1px solid rgba(255,255,255,0.08);'
                        f'border-top:2px solid #f2ca50;border-radius:12px;padding:14px 16px;margin-top:8px;">'
                        f'<div style="font-size:.7rem;color:#d6e4f7;opacity:.6;text-transform:uppercase;letter-spacing:.05em;">Top Driver</div>'
                        f'<div style="font-size:1.2rem;font-weight:700;color:#f2ca50;">{_top_feat}</div>'
                        f'<div style="font-size:.8rem;color:#d6e4f7;opacity:.7;">Category: {_top_cat} · {_top_pct:.1f}% of total</div>'
                        f'</div>',
                        unsafe_allow_html=True)


            # ── SHAP Explainability ───────────────────────────────────────
            if model_name in ("Random Forest", "XGBoost", "LightGBM") and hasattr(mdl, "feature_importances_"):
                with st.expander("SHAP Explainability — Why did the model signal Buy/Sell?", expanded=False):
                    try:
                        import shap
                        shap.initjs()
                        _shap_explainer = shap.TreeExplainer(mdl)
                        _shap_vals = _shap_explainer.shap_values(X_te[:min(200, len(X_te))])
                        _shap_df   = pd.DataFrame(_shap_vals, columns=feature_cols)

                        # Mean |SHAP| per feature
                        _mean_shap = _shap_df.abs().mean().sort_values(ascending=False)
                        _top_shap  = _mean_shap.head(15)

                        sh_c1, sh_c2 = st.columns([3, 2])
                        with sh_c1:
                            st.markdown("**Mean |SHAP| — Average impact on model output**")
                            _shap_colors = ["#22c55e" if _shap_df[f].mean() > 0 else "#ef4444"
                                            for f in _top_shap.index]
                            fig_shap = go.Figure(go.Bar(
                                x=_top_shap.values, y=_top_shap.index,
                                orientation="h", marker_color=_shap_colors,
                                text=[f"{v:.4f}" for v in _top_shap.values],
                                textposition="outside"))
                            fig_shap.update_layout(
                                height=400, template="plotly_dark",
                                paper_bgcolor="#0d1b2a", plot_bgcolor="#061422",
                                xaxis_title="Mean |SHAP value|",
                                margin=dict(l=0, r=40, t=10, b=0))
                            st.plotly_chart(fig_shap, width="stretch")

                        with sh_c2:
                            st.markdown("**Last Signal Explanation**")
                            _last_shap = pd.Series(_shap_explainer.shap_values(X_te[-1:])[0],
                                                   index=feature_cols).sort_values(key=abs, ascending=False).head(8)
                            _last_pred = float(preds_ret[-1])
                            _sig_color = "#22c55e" if _last_pred > 0.15 else "#ef4444" if _last_pred < -0.15 else "#f2ca50"
                            st.markdown(
                                f'<div style="background:#13212e;border:1px solid rgba(255,255,255,0.08);'
                                f'border-radius:8px;padding:12px 14px;margin-bottom:10px;">'
                                f'<div style="font-size:.7rem;color:#d6e4f7;opacity:.6;text-transform:uppercase;">Last prediction</div>'
                                f'<div style="font-size:1.6rem;font-weight:800;color:{_sig_color};">{_last_pred:+.3f}%</div>'
                                f'<div style="font-size:.75rem;color:#d6e4f7;opacity:.5;">Top contributing features (SHAP):</div></div>',
                                unsafe_allow_html=True)
                            for feat, val in _last_shap.items():
                                bar_c = "#22c55e" if val > 0 else "#ef4444"
                                st.markdown(
                                    f'<div style="display:flex;justify-content:space-between;'
                                    f'background:#0d1b2a;border-radius:6px;padding:5px 10px;margin:2px 0;">'
                                    f'<span style="font-size:.78rem;color:#d6e4f7;">{feat}</span>'
                                    f'<span style="font-size:.78rem;font-weight:700;color:{bar_c};">{val:+.4f}</span></div>',
                                    unsafe_allow_html=True)
                            st.caption("Green = pushed prediction UP (bullish) · Red = pushed DOWN (bearish)")
                    except ImportError:
                        st.info("Install SHAP for explainability: `pip install shap`")
                    except Exception as e:
                        st.warning(f"SHAP unavailable: {e}")

            with st.expander("Residual Analysis"):
                residuals = y_te - preds_ret
                fig_res   = px.histogram(residuals, nbins=60,
                                          color_discrete_sequence=['#1e2b39'],
                                          labels={'value': 'Residual (% return)'})
                fig_res.update_layout(height=250, template='plotly_dark',
                                      paper_bgcolor='#0d1b2a', plot_bgcolor='#061422',
                                      xaxis_title="Residual (% return)", showlegend=False)
                st.plotly_chart(fig_res, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Forecast":

    page_header("Gold Price Scenario Analysis",
                "Probabilistic scenario estimates — NOT price predictions. Financial markets are partially stochastic. Use as directional tendency only.")

    st.markdown(
        '<div style="background:rgba(242,202,80,0.05);border:1px solid rgba(242,202,80,0.2);'
        'border-left:3px solid #f2ca50;border-radius:8px;padding:8px 14px;'
        'font-size:0.78rem;color:#d6e4f7;opacity:0.8;margin-bottom:16px;">'
        '<b style="color:#f2ca50;">Research Disclaimer:</b> This tool estimates short-term market '
        'tendencies using historical & macro-financial indicators. Financial markets contain '
        'significant stochastic components — outputs should not be used as investment advice.'
        '</div>',
        unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        fc_model = st.selectbox("Model",
                                 ["Random Forest", "XGBoost", "LightGBM", "Prophet", "Linear Regression"],
                                 key="fc_model",
                                 help="Prophet is purpose-built for time series trends. Tree models use lag features recursively.")
    with fc2:
        fc_lags = st.slider("Lag Features (days)", 5, 60, 20, key="fc_lags",
                           help="Number of past days fed as input. Prophet ignores this — it uses the full time series.")
    with fc3:
        _horizon_map = {"1 Day":1,"1 Week":5,"1 Month":30,"3 Months":63}
        _hz_choice   = st.selectbox("Forecast Horizon",
                                    list(_horizon_map.keys()), index=2,
                                    key="fc_hz",
                                    help="Longer horizons = much wider uncertainty bands. Beyond 14 days treat as directional tendency only.")
        fc_days = _horizon_map[_hz_choice]

    fc_feats = st.multiselect(
        "Additional Features (last known values carried forward)",
        ['Price_Oil', 'Price_Dollar', 'Price_Stocks'],
        default=[], key="fc_feats")

    run_fc = st.button("Generate Forecast", type="primary", width='stretch')

    if run_fc:
        with st.spinner("Training model and generating forecast..."):

            from sklearn.model_selection import TimeSeriesSplit

            use_cols = ['Date', 'Price_Gold'] + fc_feats
            fc_df    = df[use_cols].dropna().sort_values('Date').reset_index(drop=True)

            # Returns-based features (stationary)
            fc_df['Return_Gold'] = fc_df['Price_Gold'].pct_change() * 100
            ret_extra_cols = []
            for c in fc_feats:
                if c in fc_df.columns:
                    fc_df[f'{c}_ret'] = fc_df[c].pct_change() * 100
                    ret_extra_cols.append(f'{c}_ret')

            # 1. Lag features
            for lag in range(1, fc_lags + 1):
                fc_df[f'lag_{lag}'] = fc_df['Return_Gold'].shift(lag)
            for lag in [7, 14, 21]:
                if lag > fc_lags:
                    fc_df[f'lag_{lag}'] = fc_df['Return_Gold'].shift(lag)

            # 2. Rolling statistics
            for _w in [5, 10, 14, 20, 30]:
                fc_df[f'rolling_mean_{_w}'] = fc_df['Return_Gold'].rolling(_w).mean()
                fc_df[f'rolling_std_{_w}']  = fc_df['Return_Gold'].rolling(_w).std()

            # 3. Volatility features
            fc_df['vol_5']    = fc_df['Return_Gold'].rolling(5).std()  * np.sqrt(252)
            fc_df['vol_21']   = fc_df['Return_Gold'].rolling(21).std() * np.sqrt(252)
            fc_df['vol_63']   = fc_df['Return_Gold'].rolling(63).std() * np.sqrt(252)
            fc_df['vol_ratio']= fc_df['vol_5'] / (fc_df['vol_63'] + 1e-9)

            # 4. Momentum features
            for _d in [5, 7, 10, 14, 21, 30]:
                fc_df[f'mom_{_d}d'] = fc_df['Price_Gold'].pct_change(_d) * 100
            fc_df['roc_5']    = fc_df['Price_Gold'].pct_change(5)  * 100
            fc_df['roc_20']   = fc_df['Price_Gold'].pct_change(20) * 100
            _sma20_fc = fc_df['Price_Gold'].rolling(20).mean()
            _std20_fc = fc_df['Price_Gold'].rolling(20).std()
            fc_df['zscore_20'] = (fc_df['Price_Gold'] - _sma20_fc) / (_std20_fc + 1e-9)

            fc_df = fc_df.dropna().reset_index(drop=True)

            _fc_fixed_lags = [f'lag_{l}' for l in [7,14,21] if l > fc_lags]
            _fc_roll = [f'rolling_mean_{w}' for w in [5,10,14,20,30]] + \
                       [f'rolling_std_{w}'  for w in [5,10,14,20,30]]
            _fc_vol  = ['vol_5','vol_21','vol_63','vol_ratio']
            _fc_mom  = [f'mom_{d}d' for d in [5,7,10,14,21,30]] + ['roc_5','roc_20','zscore_20']
            ts_feat_cols = _fc_roll + _fc_vol + _fc_mom
            feature_cols = ([f'lag_{i}' for i in range(1, fc_lags + 1)] +
                            _fc_fixed_lags + ret_extra_cols + ts_feat_cols)
            feature_cols = [c for c in feature_cols if c in fc_df.columns]

            X_all      = fc_df[feature_cols].values
            y_all      = fc_df['Return_Gold'].values
            prices_arr = fc_df['Price_Gold'].values

            if fc_model == "Prophet":
                try:
                    from prophet import Prophet as _Prophet
                    import warnings
                    warnings.filterwarnings("ignore")
                    _prop_df = pd.DataFrame({
                        'ds': fc_df['Date'],
                        'y':  np.log(fc_df['Price_Gold'])  # log for stability
                    })
                    _m = _Prophet(
                        daily_seasonality=False,
                        weekly_seasonality=True,
                        yearly_seasonality=True,
                        changepoint_prior_scale=0.15,
                        seasonality_prior_scale=10,
                    )
                    _m.fit(_prop_df)
                    _future   = _m.make_future_dataframe(periods=fc_days + 30, freq='D')
                    _fcst     = _m.predict(_future)
                    # keep only future business days
                    _fcst_fut = _fcst[_fcst['ds'] > fc_df['Date'].iloc[-1]].copy()
                    _fcst_fut = _fcst_fut[_fcst_fut['ds'].dt.weekday < 5].head(fc_days)
                    forecast_dates  = list(_fcst_fut['ds'])
                    forecast_prices = list(np.exp(_fcst_fut['yhat'].values))
                    _lo = np.exp(_fcst_fut['yhat_lower'].values)
                    _hi = np.exp(_fcst_fut['yhat_upper'].values)
                    upper_band = _hi
                    lower_band = _lo
                    cv_rmse    = float(np.mean(_hi - _lo)) / 4
                except ImportError:
                    st.error("Prophet not installed. Run: pip install prophet")
                    st.stop()
            else:
                if fc_model == "Random Forest":
                    mdl = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
                elif fc_model == "XGBoost":
                    try:
                        from xgboost import XGBRegressor
                        mdl = XGBRegressor(n_estimators=300, learning_rate=0.05,
                                           max_depth=6, random_state=42, verbosity=0)
                    except ImportError:
                        st.error("XGBoost not available. Choose another model.")
                        st.stop()
                elif fc_model == "LightGBM":
                    try:
                        from lightgbm import LGBMRegressor
                        mdl = LGBMRegressor(n_estimators=500, learning_rate=0.03,
                                            num_leaves=63, min_child_samples=20,
                                            subsample=0.8, colsample_bytree=0.8,
                                            random_state=42, verbose=-1)
                    except ImportError:
                        st.error("LightGBM not available. Choose another model.")
                        st.stop()
                else:
                    mdl = LinearRegression()

            if fc_model != "Prophet":
                tscv      = TimeSeriesSplit(n_splits=5)
                cv_errors = []
                for tr_idx, te_idx in tscv.split(X_all):
                    mdl.fit(X_all[tr_idx], y_all[tr_idx])
                    preds_cv = mdl.predict(X_all[te_idx])
                    for local_i, global_i in enumerate(te_idx):
                        if global_i > 0:
                            p_pred = prices_arr[global_i-1] * (1 + preds_cv[local_i] / 100)
                            cv_errors.append(abs(prices_arr[global_i] - p_pred))
                cv_rmse = float(np.mean(cv_errors)) if cv_errors else 50.0

                mdl.fit(X_all, y_all)

                ret_hist   = list(fc_df['Return_Gold'].values)
                price_hist = list(fc_df['Price_Gold'].values)
                last_extra_rets = {f'{c}_ret': fc_df[f'{c}_ret'].iloc[-1]
                                   for c in fc_feats if f'{c}_ret' in fc_df.columns}
                last_date  = fc_df['Date'].iloc[-1]

                forecast_dates, forecast_prices = [], []
                for step in range(fc_days):
                    lags_row  = [ret_hist[-(i)] for i in range(1, fc_lags + 1)]
                    extra_row = [last_extra_rets.get(f'{c}_ret', 0) for c in fc_feats]
                    _ts_vals  = [fc_df[c].iloc[-1] for c in ts_feat_cols if c in fc_df.columns]
                    pred_ret  = float(mdl.predict(
                        np.array(lags_row + extra_row + _ts_vals).reshape(1, -1))[0])
                    pred_ret  = float(np.clip(pred_ret, -5.0, 5.0))
                    pred_price = price_hist[-1] * (1 + pred_ret / 100)
                    next_date  = last_date + pd.Timedelta(days=step + 1)
                    while next_date.weekday() >= 5:
                        next_date += pd.Timedelta(days=1)
                    forecast_dates.append(next_date)
                    forecast_prices.append(pred_price)
                    ret_hist.append(pred_ret)
                    price_hist.append(pred_price)
                    last_date = next_date

            forecast_prices = np.array(forecast_prices)

            # ── Monte Carlo Probabilistic Forecast (500 paths) ───────────
            N_SIM = 500
            if fc_model != "Prophet":
                # Residuals from CV as empirical noise distribution
                _resid_src = fc_df["Return_Gold"].diff().dropna().values
                _resid_std = float(np.std(np.array(cv_errors) / (fc_df["Price_Gold"].mean() + 1e-9) * 100)) \
                             if cv_errors else 0.5
                rng = np.random.default_rng(42)
                sim_paths = np.zeros((N_SIM, fc_days))
                for _s in range(N_SIM):
                    _ph = [float(fc_df["Price_Gold"].iloc[-1])]
                    _rh = list(fc_df["Return_Gold"].values)
                    for _step in range(fc_days):
                        # predicted return + bootstrap residual noise
                        _noise = rng.choice(_resid_src) if len(_resid_src) > 0 else \
                                 rng.normal(0, _resid_std)
                        _sim_ret = float(np.clip(
                            rng.normal(forecast_prices[_step] / _ph[-1] * 100 - 100,
                                       abs(_noise) * 0.5), -8, 8))
                        _sim_p   = _ph[-1] * (1 + _sim_ret / 100)
                        sim_paths[_s, _step] = _sim_p
                        _ph.append(_sim_p)
                p10 = np.percentile(sim_paths, 10, axis=0)
                p25 = np.percentile(sim_paths, 25, axis=0)
                p50 = np.percentile(sim_paths, 50, axis=0)
                p75 = np.percentile(sim_paths, 75, axis=0)
                p90 = np.percentile(sim_paths, 90, axis=0)
            else:
                p10, p25 = np.array(lower_band), np.array(lower_band)
                p50 = forecast_prices
                p75, p90 = np.array(upper_band), np.array(upper_band)
                sim_paths = None

            last_actual   = float(fc_df["Price_Gold"].iloc[-1])
            last_forecast = float(p50[-1])
            change_usd    = last_forecast - last_actual
            change_pct    = (change_usd / last_actual) * 100

            # ── KPI Cards ────────────────────────────────────────────────
            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("Last Actual Price",        f"${last_actual:,.2f}")
            sm2.metric(f"Day {fc_days} Base Scenario",    f"${last_forecast:,.2f}",
                       f"{change_usd:+,.2f} ({change_pct:+.1f}%)")
            sm3.metric(f"80% Range (Day {fc_days})",
                       f"${p10[-1]:,.0f} – ${p90[-1]:,.0f}")
            sm4.metric(f"50% Range (Day {fc_days})",
                       f"${p25[-1]:,.0f} – ${p75[-1]:,.0f}")

            # ── Probability summary card ─────────────────────────────────
            _prob_up = 0.0
            if sim_paths is not None:
                _prob_up = float(np.mean(sim_paths[:, -1] > last_actual) * 100)
            _dir_txt = "Bullish" if _prob_up >= 55 else ("Bearish" if _prob_up <= 45 else "Neutral")
            _dir_col = "#22c55e" if _prob_up >= 55 else ("#ef4444" if _prob_up <= 45 else "#f2ca50")
            _card_html = (
                f'<div style="background:#13212e;border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:14px 20px;margin:8px 0;">'
                f'<span style="font-size:0.75rem;color:#d6e4f7;opacity:.6;text-transform:uppercase;letter-spacing:.05em;">Probability Price Higher in {fc_days}d</span><br>'
                f'<span style="font-size:2rem;font-weight:700;color:{_dir_col};">{_prob_up:.0f}%</span>&nbsp;'
                f'<span style="color:{_dir_col};font-size:1rem;font-weight:600;">{_dir_txt}</span>'
                f'<hr style="border-color:rgba(255,255,255,0.07);margin:8px 0;">'
                f'<span style="font-size:0.82rem;color:#d6e4f7;opacity:.7;">Based on {N_SIM} Monte Carlo simulations &nbsp;·&nbsp; '
                f'50% range: <b style="color:#f2ca50">${p25[-1]:,.0f}–${p75[-1]:,.0f}</b> &nbsp;·&nbsp; '
                f'80% range: <b style="color:#d6e4f7">${p10[-1]:,.0f}–${p90[-1]:,.0f}</b></span>'
                f'</div>'
            )
            st.markdown(_card_html, unsafe_allow_html=True)

            st.markdown("---")

            # ── Fan Chart ────────────────────────────────────────────────
            hist   = fc_df[["Date", "Price_Gold"]].tail(180)
            fig_fc = go.Figure()

            # 80% band (P10-P90)
            fig_fc.add_trace(go.Scatter(
                x=list(forecast_dates) + list(forecast_dates[::-1]),
                y=list(p90) + list(p10[::-1]),
                fill="toself", fillcolor="rgba(242,202,80,0.07)",
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip", showlegend=True, name="80% Probability Range"))
            # 50% band (P25-P75)
            fig_fc.add_trace(go.Scatter(
                x=list(forecast_dates) + list(forecast_dates[::-1]),
                y=list(p75) + list(p25[::-1]),
                fill="toself", fillcolor="rgba(242,202,80,0.18)",
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip", showlegend=True, name="50% Probability Range"))
            # Historical
            fig_fc.add_trace(go.Scatter(
                x=hist["Date"], y=hist["Price_Gold"],
                line=dict(color="#d6e4f7", width=2), name="Historical"))
            # Median forecast
            fig_fc.add_trace(go.Scatter(
                x=forecast_dates, y=p50,
                line=dict(color="#f2ca50", width=2.5, dash="dash"),
                name="Base Scenario (Median)",
                hovertemplate="%{x|%b %d, %Y}<br>Median: $%{y:,.2f}<extra></extra>"))
            # P10 / P90 lines
            fig_fc.add_trace(go.Scatter(
                x=forecast_dates, y=p10,
                line=dict(color="rgba(239,68,68,0.5)", width=1, dash="dot"),
                name="Bear Scenario (P10)", hovertemplate="$%{y:,.2f}"))
            fig_fc.add_trace(go.Scatter(
                x=forecast_dates, y=p90,
                line=dict(color="rgba(34,197,94,0.5)", width=1, dash="dot"),
                name="Bull Scenario (P90)", hovertemplate="$%{y:,.2f}"))
            _vline_x = str(fc_df["Date"].iloc[-1].date())
            fig_fc.add_shape(type="line",
                              x0=_vline_x, x1=_vline_x, y0=0, y1=1,
                              xref="x", yref="paper",
                              line=dict(color="#293644", width=1.5, dash="dot"))
            fig_fc.add_annotation(x=_vline_x, y=0.98, xref="x", yref="paper",
                                   text="Forecast Start", showarrow=False,
                                   font=dict(color="#d6e4f7", size=11), xanchor="left")
            fig_fc.update_layout(
                height=520, template="plotly_dark",
                paper_bgcolor="#0d1b2a", plot_bgcolor="#061422",
                yaxis_title="Price (USD)", hovermode="x unified",
                legend=dict(orientation="h", y=1.02, font=dict(size=10)),
                margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_fc, width="stretch")

            st.subheader("Daily Forecast Table")
            fc_table = pd.DataFrame({
                "Date":           [d.strftime("%a, %b %d %Y") for d in forecast_dates],
                "Base Scenario ($)":     [f"${p:,.2f}" for p in p50],
                "Bear Scenario ($)":   [f"${p:,.2f}" for p in p10],
                "Bull Scenario ($)":   [f"${p:,.2f}" for p in p90],
                "50% Probability Range":      [f"${a:,.0f}–${b:,.0f}" for a,b in zip(p25,p75)],
                "Δ vs Today":     [f"{((p-last_actual)/last_actual*100):+.2f}%" for p in p50],
            })
            fc_table.index = range(1, fc_days + 1)
            fc_table.index.name = "Day"
            st.dataframe(fc_table, width="stretch")
            st.download_button(
                label="Download Scenario Analysis as CSV",
                data=fc_table.to_csv().encode("utf-8"),
                file_name=f"gold_forecast_{fc_days}days_{fc_model.replace(' ','_')}.csv",
                mime="text/csv", use_container_width=True)

            with st.expander("Model Details"):
                st.markdown(f"""
- **Model:** {fc_model}
- **Training rows:** {len(X_all):,}
- **Lag features:** {fc_lags} days
- **Simulations:** {N_SIM} Monte Carlo paths
- **Method:** Recursive prediction + bootstrap residual noise per step
- **P50:** median of all simulations · **P10/P90:** bear/bull scenarios
                """)
    else:
        st.info("Configure the settings above and click **Generate Forecast** to run.")
        st.markdown("""
        <div style="background:#13212e; border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:20px; margin-top:12px;">
            <h3 style="color:#f2ca50; margin-top:0;">How It Works</h3>
            <ol style="color:#d6e4f7; line-height:2; padding-left:20px;">
                <li>Train the model on <b>all available historical data</b></li>
                <li>Use the last N actual prices as lag input features</li>
                <li>Estimate Day 1 scenario — feed that into Day 2, and so on</li>
                <li>Confidence band widens over the horizon to reflect growing uncertainty</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ABOUT
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Portfolio":

    page_header("Portfolio Optimization",
                "Modern Portfolio Theory — find the optimal allocation between Gold, Oil, S&P 500 and Dollar Index.")

    from scipy.optimize import minimize

    po1, po2 = st.columns(2)
    with po1:
        po_start = st.date_input("From", value=pd.to_datetime("2015-01-01").date(),
                                 min_value=df["Date"].min().date(),
                                 max_value=df["Date"].max().date(), key="po_from")
    with po2:
        po_end   = st.date_input("To",   value=df["Date"].max().date(),
                                 min_value=df["Date"].min().date(),
                                 max_value=df["Date"].max().date(), key="po_to")

    po_assets = st.multiselect("Assets to include",
                               ["Gold","Oil","S&P 500","Dollar Index"],
                               default=["Gold","Oil","S&P 500"], key="po_assets")

    n_sim = st.slider("Monte Carlo simulations", 500, 5000, 2000, step=500,
                      help="More simulations = smoother efficient frontier but slower")

    run_po = st.button("Optimize Portfolio", type="primary", use_container_width=True)

    if run_po:
        _acols = {"Gold":"Price_Gold","Oil":"Price_Oil",
                  "S&P 500":"Price_Stocks","Dollar Index":"Price_Dollar"}
        _mask  = (df["Date"].dt.date >= po_start) & (df["Date"].dt.date <= po_end)
        _pdf   = df[_mask][["Date"] + [_acols[a] for a in po_assets if _acols[a] in df.columns]]
        _pdf   = _pdf.dropna().sort_values("Date").reset_index(drop=True)

        if len(_pdf) < 60:
            st.warning("Need at least 60 data points. Expand the date range.")
            st.stop()

        # Daily returns
        _rets = _pdf[[_acols[a] for a in po_assets]].pct_change().dropna()
        _rets.columns = po_assets
        _mu   = _rets.mean() * 252          # annualised expected returns
        _cov  = _rets.cov()  * 252          # annualised covariance

        n_assets = len(po_assets)
        rf_rate  = 0.04                     # risk-free rate 4%

        def _portfolio_stats(w):
            w  = np.array(w)
            pr = float(w @ _mu)
            pv = float(np.sqrt(w @ _cov.values @ w))
            sr = (pr - rf_rate) / pv if pv > 1e-9 else 0
            return pr, pv, sr

        # ── Monte Carlo frontier ──────────────────────────────────────────────
        rng = np.random.default_rng(42)
        mc_results = []
        for _ in range(n_sim):
            w = rng.random(n_assets)
            w = w / w.sum()
            r, v, s = _portfolio_stats(w)
            mc_results.append({"Return": r*100, "Volatility": v*100,
                                "Sharpe": s, "Weights": w.tolist()})
        mc_df = pd.DataFrame(mc_results)

        # ── Optimised portfolios ──────────────────────────────────────────────
        constraints = ({"type":"eq","fun": lambda w: np.sum(w)-1},)
        bounds      = tuple((0.0, 1.0) for _ in range(n_assets))
        w0          = np.ones(n_assets) / n_assets

        # Max Sharpe
        res_sr = minimize(lambda w: -_portfolio_stats(w)[2],
                          w0, method="SLSQP", bounds=bounds, constraints=constraints)
        w_sr   = res_sr.x
        r_sr, v_sr, s_sr = _portfolio_stats(w_sr)

        # Min Volatility
        res_mv = minimize(lambda w: _portfolio_stats(w)[1],
                          w0, method="SLSQP", bounds=bounds, constraints=constraints)
        w_mv   = res_mv.x
        r_mv, v_mv, s_mv = _portfolio_stats(w_mv)

        # Equal weight
        w_eq = np.ones(n_assets) / n_assets
        r_eq, v_eq, s_eq = _portfolio_stats(w_eq)

        # ── KPI ──────────────────────────────────────────────────────────────
        st.markdown("---")
        pk1, pk2, pk3 = st.columns(3)
        for _col, _lbl, _w, _r, _v, _s, _c in [
            (pk1, "Max Sharpe Portfolio",    w_sr, r_sr, v_sr, s_sr, "#f2ca50"),
            (pk2, "Min Volatility Portfolio", w_mv, r_mv, v_mv, s_mv, "#22c55e"),
            (pk3, "Equal Weight",             w_eq, r_eq, v_eq, s_eq, "#60a5fa"),
        ]:
            alloc_str = " · ".join(f"{a}: {wi*100:.0f}%" for a,wi in zip(po_assets,_w))
            _col.markdown(
                f'<div style="background:#13212e;border:1px solid rgba(255,255,255,0.08);'
                f'border-top:2px solid {_c};border-radius:12px;padding:16px 18px;">'
                f'<div style="font-size:.7rem;color:#d6e4f7;opacity:.6;text-transform:uppercase;letter-spacing:.05em;">{_lbl}</div>'
                f'<div style="font-size:1.1rem;font-weight:700;color:{_c};margin:6px 0 2px;">'
                f'Return {_r*100:.1f}% · Vol {_v*100:.1f}% · Sharpe {_s:.2f}</div>'
                f'<div style="font-size:.72rem;color:#d6e4f7;opacity:.5;">{alloc_str}</div></div>',
                unsafe_allow_html=True)

        st.markdown("---")

        # ── Efficient Frontier chart ──────────────────────────────────────────
        fig_ef = go.Figure()
        fig_ef.add_trace(go.Scatter(
            x=mc_df["Volatility"], y=mc_df["Return"],
            mode="markers",
            marker=dict(color=mc_df["Sharpe"], colorscale="Viridis",
                        size=3, opacity=0.5, showscale=True,
                        colorbar=dict(title="Sharpe", thickness=12)),
            name="Simulated Portfolios",
            hovertemplate="Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>"))

        for _lbl, _w, _r, _v, _c, _sym in [
            ("Max Sharpe",    w_sr, r_sr, v_sr, "#f2ca50", "star"),
            ("Min Vol",       w_mv, r_mv, v_mv, "#22c55e", "diamond"),
            ("Equal Weight",  w_eq, r_eq, v_eq, "#60a5fa", "circle"),
        ]:
            fig_ef.add_trace(go.Scatter(
                x=[_v*100], y=[_r*100], mode="markers+text",
                marker=dict(color=_c, size=14, symbol=_sym,
                            line=dict(color="white", width=1.5)),
                text=[_lbl], textposition="top center",
                textfont=dict(color=_c, size=11),
                name=_lbl,
                hovertemplate=f"{_lbl}<br>Vol: {_v*100:.1f}%<br>Return: {_r*100:.1f}%<br>Sharpe: {_s:.2f}<extra></extra>"))

        fig_ef.update_layout(
            height=480, template="plotly_dark",
            paper_bgcolor="#0d1b2a", plot_bgcolor="#061422",
            xaxis_title="Annual Volatility (%)",
            yaxis_title="Expected Annual Return (%)",
            title="Efficient Frontier — Modern Portfolio Theory",
            hovermode="closest",
            legend=dict(orientation="h", y=1.02, font=dict(size=10)),
            margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_ef, width="stretch")

        # ── Allocation charts ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("**Portfolio Allocations**")
        al1, al2 = st.columns(2)
        for _col, _lbl, _w, _c in [
            (al1, "Max Sharpe", w_sr, "#f2ca50"),
            (al2, "Min Volatility", w_mv, "#22c55e")
        ]:
            with _col:
                fig_al = go.Figure(go.Pie(
                    labels=po_assets, values=[wi*100 for wi in _w],
                    hole=0.45,
                    marker_colors=["#f2ca50","#ef4444","#22c55e","#60a5fa"][:n_assets],
                    textinfo="label+percent"))
                fig_al.update_layout(
                    height=280, template="plotly_dark",
                    paper_bgcolor="#0d1b2a", showlegend=False,
                    title=dict(text=_lbl, font=dict(color=_c)),
                    margin=dict(l=0,r=0,t=30,b=0))
                st.plotly_chart(fig_al, width="stretch")

        st.caption("Based on Modern Portfolio Theory (Markowitz). Past performance does not guarantee future results.")


# ══════════════════════════════════════════════════════════════════════════════
# SENTIMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Sentiment":

    page_header("Market Sentiment",
                "VADER NLP sentiment analysis on live gold news headlines from Yahoo Finance.")

    @st.cache_data(ttl=1800)
    def fetch_gold_news():
        """Fetch gold-related news via yfinance — cached 30 min."""
        try:
            import yfinance as yf
            ticker = yf.Ticker("GC=F")
            news   = ticker.news
            if not news:
                return []
            items = []
            for n in news[:30]:
                title = (n.get("title") or n.get("content",{}).get("title",""))
                pub   = n.get("providerPublishTime") or n.get("content",{}).get("pubDate","")
                url   = (n.get("link") or n.get("content",{}).get("canonicalUrl",{}).get("url","#"))
                src   = (n.get("publisher") or n.get("content",{}).get("provider",{}).get("displayName",""))
                if title:
                    items.append({"title": title, "pub": pub, "url": url, "src": src})
            return items
        except Exception as e:
            return []

    @st.cache_data(ttl=1800)
    def score_sentiment(headlines):
        """Score headlines with VADER — install via pip if missing."""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        except ImportError:
            try:
                import subprocess, sys
                subprocess.run([sys.executable,"-m","pip","install",
                                "vaderSentiment","--quiet","--break-system-packages"],
                               capture_output=True)
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            except Exception:
                return [0.0] * len(headlines)
        sia = SentimentIntensityAnalyzer()
        return [sia.polarity_scores(h)["compound"] for h in headlines]

    with st.spinner("Fetching latest gold news..."):
        news_items = fetch_gold_news()

    if not news_items:
        st.warning("Could not fetch news. Try again in a few minutes.")
    else:
        headlines = [n["title"] for n in news_items]
        scores    = score_sentiment(headlines)

        # ── Overall Sentiment Gauge ───────────────────────────────────────────
        avg_score = float(np.mean(scores)) if scores else 0.0
        if avg_score >= 0.15:
            sent_label, sent_color = "Bullish", "#22c55e"
        elif avg_score <= -0.15:
            sent_label, sent_color = "Bearish", "#ef4444"
        else:
            sent_label, sent_color = "Neutral", "#f2ca50"

        bull_pct    = float(np.mean([s > 0.05  for s in scores]) * 100)
        bear_pct    = float(np.mean([s < -0.05 for s in scores]) * 100)
        neutral_pct = 100 - bull_pct - bear_pct

        g1, g2, g3, g4 = st.columns(4)
        g1.markdown(
            f'<div style="background:#13212e;border:1px solid rgba(255,255,255,0.08);'

            f'border-top:2px solid {sent_color};border-radius:12px;padding:16px 18px;">'

            f'<div style="font-size:.7rem;color:#d6e4f7;opacity:.6;text-transform:uppercase;letter-spacing:.05em;">Overall Sentiment</div>'

            f'<div style="font-size:1.8rem;font-weight:800;color:{sent_color};">{sent_label}</div>'

            f'<div style="font-size:.8rem;color:#d6e4f7;opacity:.6;">Score: {avg_score:+.3f}</div></div>',
            unsafe_allow_html=True)
        g2.metric("Bullish Headlines",  f"{bull_pct:.0f}%")
        g3.metric("Bearish Headlines",  f"{bear_pct:.0f}%")
        g4.metric("Neutral Headlines",  f"{neutral_pct:.0f}%")

        st.markdown("---")

        # ── Sentiment Distribution Chart ──────────────────────────────────────
        sc1, sc2 = st.columns([2, 1])
        with sc1:
            st.markdown("**Headline Sentiment Scores**")
            _colors = ["#22c55e" if s > 0.05 else "#ef4444" if s < -0.05 else "#f2ca50"
                       for s in scores]
            _labels = [h[:55] + "..." if len(h) > 55 else h for h in headlines]
            fig_sent = go.Figure(go.Bar(
                x=scores, y=_labels, orientation="h",
                marker_color=_colors,
                hovertemplate="%{y}<br>Score: %{x:.3f}<extra></extra>"))
            fig_sent.add_vline(x=0, line=dict(color="rgba(255,255,255,0.3)", width=1))
            fig_sent.add_vline(x=0.05,  line=dict(color="#22c55e", width=0.8, dash="dot"))
            fig_sent.add_vline(x=-0.05, line=dict(color="#ef4444", width=0.8, dash="dot"))
            fig_sent.update_layout(
                height=max(380, len(headlines)*22), template="plotly_dark",
                paper_bgcolor="#0d1b2a", plot_bgcolor="#061422",
                xaxis=dict(title="VADER Compound Score", range=[-1.05, 1.05],
                           tickvals=[-1,-0.5,-0.05,0,0.05,0.5,1]),
                margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
            st.plotly_chart(fig_sent, width="stretch")

        with sc2:
            st.markdown("**Distribution**")
            fig_pie = go.Figure(go.Pie(
                labels=["Bullish","Neutral","Bearish"],
                values=[bull_pct, neutral_pct, bear_pct],
                marker_colors=["#22c55e","#f2ca50","#ef4444"],
                hole=0.5, textinfo="label+percent"))
            fig_pie.update_layout(
                height=300, template="plotly_dark",
                paper_bgcolor="#0d1b2a", showlegend=False,
                margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig_pie, width="stretch")

            st.markdown("**Strongest Signal**")
            _top_i = int(np.argmax(np.abs(scores)))
            _top_s = scores[_top_i]
            _top_c = "#22c55e" if _top_s > 0 else "#ef4444"
            st.markdown(
                f'<div style="background:#13212e;border:1px solid rgba(255,255,255,0.08);'

                f'border-left:3px solid {_top_c};border-radius:8px;padding:12px 14px;font-size:.8rem;color:#d6e4f7;">'

                f'<b style="color:{_top_c};">{_top_s:+.3f}</b><br>{headlines[_top_i][:80]}</div>',
                unsafe_allow_html=True)

        st.markdown("---")

        # ── News Feed ─────────────────────────────────────────────────────────
        st.markdown("**Latest Headlines**")
        for i, (item, score) in enumerate(zip(news_items, scores)):
            s_color = "#22c55e" if score > 0.05 else "#ef4444" if score < -0.05 else "#f2ca50"
            s_label = "Bullish" if score > 0.05 else "Bearish" if score < -0.05 else "Neutral"
            import datetime as _dt
            try:
                _ts = _dt.datetime.fromtimestamp(int(item["pub"])).strftime("%b %d %H:%M") \
                      if item["pub"] else ""
            except Exception:
                _ts = str(item["pub"])[:16] if item["pub"] else ""
            st.markdown(
                f'<div style="background:#13212e;border:1px solid rgba(255,255,255,0.06);'

                f'border-left:3px solid {s_color};border-radius:8px;'

                f'padding:10px 14px;margin-bottom:6px;display:flex;justify-content:space-between;align-items:center;">'

                f'<div><a href="{item["url"]}" target="_blank" '

                f'style="color:#d6e4f7;text-decoration:none;font-size:.85rem;font-weight:500;">{item["title"]}</a>'

                f'<br><span style="font-size:.72rem;color:#d6e4f7;opacity:.45;">{item["src"]} &nbsp;·&nbsp; {_ts}</span></div>'

                f'<div style="font-size:.75rem;font-weight:700;color:{s_color};'

                f'white-space:nowrap;margin-left:12px;">{s_label}<br>{score:+.2f}</div></div>',
                unsafe_allow_html=True)

        st.caption("Sentiment powered by VADER NLP. Scores: +1.0 = most positive, -1.0 = most negative. Threshold: ±0.05.")


elif page == "About":

    page_header("About This App", "Aurum Gold Intelligence — open-source gold price analytics platform.")

    st.markdown("""
    <div style="background:#13212e; border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:24px 28px; margin-bottom:20px;">
        <h3 style="color:#f2ca50; margin-top:0;">Gold Price Prediction App</h3>
        <p style="color:#d6e4f7; line-height:1.8;">
            An interactive dashboard and machine learning research tool for gold market analysis.
            using historical data from Yahoo Finance covering gold futures, crude oil,
            the US Dollar Index, and the S&P 500.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background:#13212e; border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:20px; margin-bottom:16px;">
            <h3 style="color:#f2ca50; margin-top:0;">Data</h3>
            <ul style="color:#d6e4f7; line-height:2; margin:0; padding-left:20px;">
                <li>Source: Yahoo Finance (yfinance)</li>
                <li>Gold Futures (GC=F)</li>
                <li>Crude Oil Futures (CL=F)</li>
                <li>US Dollar Index (DX-Y.NYB)</li>
                <li>S&P 500 Index (^GSPC)</li>
                <li>Updated daily via GitHub Actions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#13212e; border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:20px;">
            <h3 style="color:#f2ca50; margin-top:0;">Dashboard Features</h3>
            <ul style="color:#d6e4f7; line-height:2; margin:0; padding-left:20px;">
                <li>Candlestick and Line charts</li>
                <li>RSI and Bollinger Bands</li>
                <li>Asset comparison (normalized)</li>
                <li>Correlation heatmap</li>
                <li>Yearly average and volume charts</li>
                <li>Investment Simulator with drawdown</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:#13212e; border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:20px; margin-bottom:16px;">
            <h3 style="color:#f2ca50; margin-top:0;">ML Models (Prediction)</h3>
            <ul style="color:#d6e4f7; line-height:2; margin:0; padding-left:20px;">
                <li><b style="color:#f2ca50;">Random Forest</b> — ensemble of decision trees</li>
                <li><b style="color:#f2ca50;">XGBoost</b> — extreme gradient boosting</li>
                <li><b style="color:#f2ca50;">LightGBM</b> — fast gradient boosting, less overfitting</li>
                <li><b style="color:#f2ca50;">Neural Network (MLP)</b> — multi-layer perceptron</li>
                <li><b style="color:#f2ca50;">Linear Regression</b> — baseline model</li>
            </ul>
        </div>
        <div style="background:#13212e; border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:20px;">
            <h3 style="color:#f2ca50; margin-top:0;">ML Models (Forecast)</h3>
            <ul style="color:#d6e4f7; line-height:2; margin:0; padding-left:20px;">
                <li><b style="color:#f2ca50;">Prophet</b> — Meta's time-series model, handles trend &amp; seasonality</li>
                <li><b style="color:#f2ca50;">LightGBM / XGBoost / RF</b> — recursive lag-feature forecast</li>
                <li><b style="color:#f2ca50;">Linear Regression</b> — baseline</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#13212e; border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:20px;">
            <h3 style="color:#f2ca50; margin-top:0;">Tech Stack</h3>
            <ul style="color:#d6e4f7; line-height:2; margin:0; padding-left:20px;">
                <li>Python 3.11 + Streamlit</li>
                <li>Plotly (interactive charts)</li>
                <li>scikit-learn / XGBoost / LightGBM / Prophet</li>
                <li>pandas / numpy</li>
                <li>GitHub Actions (auto daily update)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#13212e; border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:20px; margin-top:4px;">
        <p style="color:#d6e4f7; margin:0; font-size:0.85rem; opacity:0.7; text-align:center;">
            Built with Streamlit · Data from Yahoo Finance · Auto-updated daily
        </p>
    </div>
    """, unsafe_allow_html=True)
