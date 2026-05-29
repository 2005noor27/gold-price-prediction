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
.stApp { background-color: #09090b; }
[data-testid="stSidebar"] { background-color: #0d0d10; border-right: 1px solid #2e5f65; }
h1 { color: #ffc72c !important; letter-spacing: 1px; }
h2, h3 { color: #ffc72c !important; }
[data-testid="stMetric"] {
    background: #1c1f23;
    border: 1px solid #2e5f65;
    border-top: 3px solid #ffc72c;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricValue"] { font-size: 1.5rem; color: #ffc72c !important; font-weight: 700; }
[data-testid="stMetricLabel"] { font-size: 0.82rem; color: #e0f7fa; opacity: 0.7; }
[data-testid="stMetricDelta"] { font-size: 0.85rem; }
.stButton > button {
    background: #ffc72c; color: #09090b; font-weight: 800;
    border: none; border-radius: 8px; padding: 10px 24px;
    transition: all 0.2s ease; letter-spacing: 0.5px;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); box-shadow: 0 4px 15px rgba(255,199,44,0.35); }
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stMultiSelect"] label { color: #e0f7fa !important; font-size: 0.9rem; opacity: 0.8; }
hr { border-color: #2e5f65; }
[data-testid="stExpander"] { border: 1px solid #2e5f65; border-radius: 8px; background: #1c1f23; }
.stRadio label { color: #e0f7fa !important; }
.block-container { padding-top: 1.5rem; }
[data-testid="stAlert"] { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


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
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:16px 0 8px 0;">
        <svg viewBox="0 0 120 120" width="88" height="88" xmlns="http://www.w3.org/2000/svg" style="display:block;margin:0 auto 6px auto;">
          <defs>
            <radialGradient id="cg" cx="38%" cy="35%" r="65%">
              <stop offset="0%"   stop-color="#ffe566"/>
              <stop offset="55%"  stop-color="#ffc72c"/>
              <stop offset="100%" stop-color="#b8860b"/>
            </radialGradient>
            <radialGradient id="eg" cx="40%" cy="38%" r="60%">
              <stop offset="0%"   stop-color="#ffd700"/>
              <stop offset="100%" stop-color="#b8860b"/>
            </radialGradient>
          </defs>
          <!-- outer ring / ridges -->
          <circle cx="60" cy="60" r="58" fill="#b8860b"/>
          <circle cx="60" cy="60" r="54" fill="url(#cg)"/>
          <!-- inner rim -->
          <circle cx="60" cy="60" r="50" fill="none" stroke="#e6a800" stroke-width="1.5" opacity="0.6"/>
          <!-- coin face -->
          <circle cx="60" cy="60" r="46" fill="url(#eg)"/>
          <!-- $ sign shadow/depth -->
          <text x="63" y="80" text-anchor="middle" font-family="Georgia,serif" font-size="54" font-weight="900"
                fill="#b8860b" opacity="0.45">$</text>
          <!-- $ sign -->
          <text x="60" y="78" text-anchor="middle" font-family="Georgia,serif" font-size="54" font-weight="900"
                fill="#ffc72c">$</text>
          <!-- shine -->
          <ellipse cx="44" cy="38" rx="12" ry="6" fill="white" opacity="0.18" transform="rotate(-30 44 38)"/>
        </svg>
        <div style="font-size:1.25rem; font-weight:800; color:#ffc72c; letter-spacing:1px;">Gold Price</div>
        <div style="font-size:0.78rem; color:#e0f7fa; opacity:0.7; margin-top:2px; letter-spacing:2px;">PREDICTION APP</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#2e5f65; margin:8px 0 12px 0;'>", unsafe_allow_html=True)

    page = st.radio("nav", ["Home", "Dashboard", "Prediction", "Forecast", "About"],
                    label_visibility="collapsed")

    st.markdown("<hr style='border-color:#2e5f65; margin:12px 0;'>", unsafe_allow_html=True)

    if page == "Dashboard":
        st.markdown("**Price Unit**")
        unit = st.radio("unit", ["oz (Ounce)", "g (Gram)"],
                        horizontal=True, label_visibility="collapsed")
        st.markdown("<hr style='border-color:#2e5f65; margin:8px 0 12px 0;'>",
                    unsafe_allow_html=True)
    else:
        unit = "oz (Ounce)"

    if page not in ("About", "Forecast", "Home"):
        st.markdown("**Date Range**")
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        start_date = st.date_input("From", value=pd.to_datetime("2010-01-01").date(),
                                   min_value=min_date, max_value=max_date)
        end_date   = st.date_input("To",   value=max_date,
                                   min_value=min_date, max_value=max_date)
        st.markdown("<hr style='border-color:#2e5f65; margin:12px 0;'>", unsafe_allow_html=True)
    else:
        start_date = pd.to_datetime("2010-01-01").date()
        end_date   = df['Date'].max().date()

    st.markdown(f"""
    <div style="font-size:0.78rem; color:#2e5f65; line-height:2;">
        &nbsp;Source: Yahoo Finance<br>
        &nbsp;Period: 1986 – present<br>
        &nbsp;{len(df):,} trading days
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
      body { background:#09090b; font-family: sans-serif; overflow:hidden; }
      .hero {
        display:flex; flex-direction:column; align-items:center;
        justify-content:center; height:100vh; width:100%;
      }
      model-viewer {
        width:420px; height:420px; background:transparent;
        --progress-bar-color: #ffc72c;
      }
      .title {
        font-size:2.8rem; font-weight:900; color:#ffc72c;
        letter-spacing:2px; text-align:center; margin-top:8px;
        text-shadow: 0 0 40px rgba(255,199,44,0.4);
      }
      .subtitle {
        font-size:1rem; color:#e0f7fa; opacity:0.6; margin-top:10px;
        letter-spacing:3px; text-align:center; text-transform:uppercase;
      }
      .badge {
        display:inline-block; margin-top:22px;
        background:rgba(255,199,44,0.12);
        border:1px solid rgba(255,199,44,0.35);
        color:#ffc72c; padding:8px 24px; border-radius:999px;
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
    <div style="text-align:center;color:#e0f7fa;opacity:0.45;font-size:0.78rem;margin-top:4px;">
        Drag to rotate &nbsp;&middot;&nbsp; Use the sidebar to navigate
    </div>
    ''', unsafe_allow_html=True)


if page == "Dashboard":

    # Unit conversion setup
    _pdiv = 31.1035 if unit == "g (Gram)" else 1.0
    _ulbl = "g" if unit == "g (Gram)" else "oz"
    _ufmt = lambda v: f"${v:,.3f}" if _ulbl == "g" else f"${v:,.2f}"
    disp_df = filtered_df.copy()
    for _c in ['Price_Gold', 'High_Gold', 'Low_Gold', 'Open_Gold']:
        if _c in disp_df.columns:
            disp_df[_c] = disp_df[_c] / _pdiv

    # Hero image + title
    hero_col, title_col = st.columns([1, 2])
    with hero_col:
        st.image("gold_hero.jpg", use_container_width=True)
    with title_col:
        st.markdown("<h1 style='margin-top:18px;'>Gold Price Dashboard</h1>", unsafe_allow_html=True)
        st.markdown(f"**{start_date}** → **{end_date}** &nbsp;|&nbsp; {len(filtered_df):,} trading days")

    # KPI Cards
    gold_clean = disp_df.dropna(subset=['Price_Gold'])
    if not gold_clean.empty:
        latest = gold_clean.iloc[-1]
        first  = gold_clean.iloc[0]
        delta  = latest['Price_Gold'] - first['Price_Gold']
        pct    = (delta / first['Price_Gold']) * 100
        c1, c2, c3, c4 = st.columns(4)
        _d_str = f"{delta:+,.3f}" if _ulbl=="g" else f"{delta:+,.2f}"
        c1.metric(f"Current Price ({_ulbl})",  _ufmt(latest['Price_Gold']), f"{_d_str} ({pct:+.1f}%)")
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
            increasing=dict(line=dict(color='#ffc72c', width=1), fillcolor='rgba(255,199,44,0.85)'),
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
                                  line=dict(color='#ffc72c', width=2),
                                  fill='tozeroy', fillcolor='rgba(255,199,44,0.08)'))
        fig.add_trace(go.Scatter(x=disp_df['Date'], y=disp_df['High_Gold'],
                                  mode='lines', name='High',
                                  line=dict(color='rgba(0,200,0,0.4)', width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=disp_df['Date'], y=disp_df['Low_Gold'],
                                  mode='lines', name='Low',
                                  line=dict(color='rgba(255,80,80,0.4)', width=1, dash='dot'),
                                  fill='tonexty', fillcolor='rgba(180,180,180,0.04)'))

    fig.update_layout(height=480, template='plotly_dark',
                      paper_bgcolor='#1c1f23', plot_bgcolor='#09090b',
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
                                     line=dict(color='#2e5f65', width=1.2, dash='dot'),
                                     name='Upper Band'))
        fig_bb.add_trace(go.Scatter(x=bb_plot['Date'], y=bb_plot['BB_lower'],
                                     line=dict(color='#2e5f65', width=1.2, dash='dot'),
                                     name='Lower Band'))
        fig_bb.add_trace(go.Scatter(x=bb_plot['Date'], y=bb_plot['SMA'],
                                     line=dict(color='#e0f7fa', width=1.5, dash='dash'),
                                     name=f'SMA {bb_period}'))
        fig_bb.add_trace(go.Scatter(x=bb_plot['Date'], y=bb_plot['Price_Gold'],
                                     line=dict(color='#ffc72c', width=2),
                                     name='Gold Price'))
        fig_bb.update_layout(height=360, template='plotly_dark',
                              paper_bgcolor='#1c1f23', plot_bgcolor='#09090b',
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
                                      line=dict(color='#ffc72c', width=2)))
        fig_rsi.update_layout(height=360, template='plotly_dark',
                               paper_bgcolor='#1c1f23', plot_bgcolor='#09090b',
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
            palette = {'Gold': '#ffc72c', 'Oil': '#2e5f65',
                       'Dollar Index': '#e0f7fa', 'S&P 500': '#4a5568'}
            for col, color in palette.items():
                fig2.add_trace(go.Scatter(x=norm['Date'], y=norm[col],
                                           mode='lines', name=col,
                                           line=dict(color=color, width=1.5)))
            fig2.update_layout(height=340, template='plotly_dark',
                               paper_bgcolor='#1c1f23', plot_bgcolor='#09090b',
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
                               paper_bgcolor='#1c1f23', plot_bgcolor='#09090b')
            st.plotly_chart(fig3, width='stretch')
        else:
            st.info("Not enough data for correlation.")

    # ── Yearly Average & Volume ───────────────────────────────────────────────
    l2, r2 = st.columns(2)

    with l2:
        st.subheader("Yearly Average Gold Price")
        yearly = disp_df.groupby('Year')['Price_Gold'].mean().reset_index()
        fig4 = px.bar(yearly, x='Year', y='Price_Gold', color='Price_Gold',
                      color_continuous_scale=[[0, '#2e5f65'], [0.5, '#ffc72c'], [1, '#ffffff']],
                      labels={'Price_Gold': f'Avg Price (USD/{_ulbl})'})
        fig4.update_layout(height=300, template='plotly_dark',
                           paper_bgcolor='#1c1f23', plot_bgcolor='#09090b',
                           showlegend=False, xaxis_title="")
        st.plotly_chart(fig4, width='stretch')

    with r2:
        st.subheader("Trading Volume")
        vol = filtered_df[['Date', 'Volume_Gold']].dropna()
        if not vol.empty:
            fig5 = go.Figure(go.Bar(x=vol['Date'], y=vol['Volume_Gold'],
                                     marker_color='rgba(255,199,44,0.7)'))
            fig5.update_layout(height=300, template='plotly_dark',
                               paper_bgcolor='#1c1f23', plot_bgcolor='#09090b',
                               yaxis_title="Volume", xaxis_title="")
            st.plotly_chart(fig5, width='stretch')
        else:
            st.info("No volume data in selected range.")

    # ── Daily Change Distribution ─────────────────────────────────────────────
    st.subheader("Daily Change % Distribution")
    chg = filtered_df['Change%_Gold'].dropna() * 100
    if not chg.empty:
        fig6 = px.histogram(chg, nbins=80, color_discrete_sequence=['#ffc72c'],
                             labels={'value': 'Daily Change (%)', 'count': 'Days'})
        fig6.update_layout(height=250, template='plotly_dark',
                           paper_bgcolor='#1c1f23', plot_bgcolor='#09090b',
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
        name='MACD', line=dict(color='#ffc72c', width=2)))
    fig_macd.add_trace(go.Scatter(
        x=macd_df['Date'], y=macd_df['Signal'],
        name='Signal', line=dict(color='#e0f7fa', width=1.5, dash='dot')))
    fig_macd.add_hline(y=0, line=dict(color='#4a5568', width=1, dash='dot'))
    fig_macd.update_layout(
        height=320, template='plotly_dark',
        paper_bgcolor='#1c1f23', plot_bgcolor='#09090b',
        yaxis_title="MACD Value", hovermode='x unified',
        legend=dict(orientation='h', y=1.02, font=dict(size=10)),
        margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_macd, width='stretch')

    # ── Investment Simulator ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Investment Simulator")
    st.markdown("See how much your investment would have grown over the selected period.")

    sim_c1, sim_c2 = st.columns([1, 2])
    with sim_c1:
        invest_amount = st.number_input("Initial Investment (USD)",
                                        min_value=100, max_value=10_000_000,
                                        value=10_000, step=1_000, key="sim_amount")
        sim_assets = st.multiselect("Compare with",
                                    ['Oil', 'S&P 500', 'Dollar Index'],
                                    default=['S&P 500'], key="sim_assets")

    asset_col_map   = {'Gold': 'Price_Gold', 'Oil': 'Price_Oil',
                       'S&P 500': 'Price_Stocks', 'Dollar Index': 'Price_Dollar'}
    selected_assets = ['Gold'] + sim_assets
    sim_cols = ['Date'] + [asset_col_map[a] for a in selected_assets
                           if asset_col_map[a] in filtered_df.columns]
    sim_df   = filtered_df[sim_cols].dropna().sort_values('Date').reset_index(drop=True)

    with sim_c1:
        if not sim_df.empty:
            sim_years = (sim_df['Date'].iloc[-1] - sim_df['Date'].iloc[0]).days / 365.25
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
                        results[asset] = {'final_val': final_val,
                                          'total_ret': total_ret, 'cagr': cagr}
            best_asset = max(results, key=lambda a: results[a]['final_val']) if results else None
            for asset, res in results.items():
                label = f"{asset} (Best)" if asset == best_asset else asset
                st.metric(label, f"${res['final_val']:,.0f}",
                          f"{res['total_ret']:+.1f}%  |  CAGR {res['cagr']:+.1f}%")

    with sim_c2:
        if not sim_df.empty:
            fig_sim  = go.Figure()
            pal_sim  = {'Gold': '#ffc72c', 'Oil': '#2e5f65',
                        'S&P 500': '#e0f7fa', 'Dollar Index': '#4a5568'}
            for asset in selected_assets:
                col = asset_col_map.get(asset)
                if col and col in sim_df.columns:
                    p0     = sim_df[col].iloc[0]
                    series = (sim_df[col] / p0) * invest_amount
                    fig_sim.add_trace(go.Scatter(
                        x=sim_df['Date'], y=series, mode='lines', name=asset,
                        line=dict(color=pal_sim.get(asset, '#ffffff'), width=2),
                        hovertemplate='%{x|%b %d, %Y}<br>$%{y:,.0f}<extra>' + asset + '</extra>'))
            fig_sim.add_hline(y=invest_amount,
                              line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dot'),
                              annotation_text=f"Initial ${invest_amount:,.0f}",
                              annotation_position="bottom right",
                              annotation=dict(font_color='rgba(255,255,255,0.4)', font_size=10))
            fig_sim.update_layout(height=340, template='plotly_dark',
                                  paper_bgcolor='#1c1f23', plot_bgcolor='#09090b',
                                  yaxis_title="Portfolio Value (USD)", hovermode='x unified',
                                  legend=dict(orientation='h', y=1.02, font=dict(size=11)),
                                  margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_sim, width='stretch')

            gold_series = (sim_df['Price_Gold'] / sim_df['Price_Gold'].iloc[0]) * invest_amount
            rolling_max = gold_series.cummax()
            drawdown    = (gold_series - rolling_max) / rolling_max * 100
            max_dd      = drawdown.min()
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=sim_df['Date'], y=drawdown,
                                         fill='tozeroy', fillcolor='rgba(239,68,68,0.15)',
                                         line=dict(color='#ef4444', width=1.2),
                                         name='Gold Drawdown',
                                         hovertemplate='%{x|%b %d, %Y}<br>%{y:.1f}%<extra></extra>'))
            fig_dd.update_layout(height=180, template='plotly_dark',
                                 paper_bgcolor='#1c1f23', plot_bgcolor='#09090b',
                                 yaxis=dict(title="Drawdown (%)", ticksuffix='%'),
                                 margin=dict(l=0, r=0, t=4, b=0), showlegend=False,
                                 annotations=[dict(x=0.01, y=0.05, xref='paper', yref='paper',
                                                   text=f"Max Drawdown: {max_dd:.1f}%",
                                                   showarrow=False,
                                                   font=dict(color='#ef4444', size=11))])
            st.plotly_chart(fig_dd, width='stretch')
        else:
            st.info("Not enough data in the selected range to simulate.")

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
# PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Prediction":

    st.markdown("<h1>Gold Price Prediction</h1>", unsafe_allow_html=True)
    st.markdown("Train a machine learning model to predict gold prices from historical patterns.")

    c1, c2, c3 = st.columns(3)
    with c1:
        model_name = st.selectbox("Model",
                                   ["Random Forest", "XGBoost", "LightGBM", "Linear Regression", "Neural Network (MLP)"])
    with c2:
        test_pct = st.slider("Test Size %", 10, 40, 20)
    with c3:
        n_lags = st.slider("Lag Features (days)", 1, 30, 5)

    _tech_opts  = [c for c in ['EMA10_pct','EMA20_pct','MACD_pct','ATR_pct'] if c in df.columns]
    extra_feats = st.multiselect(
        "Additional Features",
        ['Price_Oil', 'Price_Dollar', 'Price_Stocks'] + _tech_opts,
        default=['Price_Oil', 'Price_Dollar', 'MACD_pct', 'ATR_pct'])

    run = st.button("Train Model", type="primary", width='stretch')

    if run:
        with st.spinner("Training ... please wait"):

            cols = ['Date', 'Price_Gold'] + extra_feats
            ml   = filtered_df[cols].dropna().sort_values('Date').reset_index(drop=True)

            # ── تحويل للعوائد اليومية (stationary) ───────────────────────────
            ml['Return_Gold'] = ml['Price_Gold'].pct_change() * 100   # % يومي

            # Lag features على العوائد (ليس الأسعار)
            for lag in range(1, n_lags + 1):
                ml[f'lag_{lag}'] = ml['Return_Gold'].shift(lag)

            # Rolling statistics on returns (time-series features)
            for _w in [5, 10, 20]:
                ml[f'rolling_mean_{_w}'] = ml['Return_Gold'].rolling(_w).mean()
                ml[f'rolling_std_{_w}']  = ml['Return_Gold'].rolling(_w).std()
            ml['momentum_5']  = ml['Price_Gold'].pct_change(5)  * 100
            ml['momentum_10'] = ml['Price_Gold'].pct_change(10) * 100

            # Price assets -> returns; indicator features used directly
            price_feats = [f for f in extra_feats if f.startswith('Price_')]
            indic_feats = [f for f in extra_feats if not f.startswith('Price_')]
            for feat in price_feats:
                ml[f'{feat}_ret'] = ml[feat].pct_change() * 100

            ml = ml.dropna().reset_index(drop=True)

            ret_extra    = [f'{f}_ret' for f in price_feats if f'{f}_ret' in ml.columns] + \
                           [f for f in indic_feats if f in ml.columns]
            feature_cols = [f'lag_{i}' for i in range(1, n_lags + 1)] + ret_extra
            feature_cols += [c for c in [
                'rolling_mean_5','rolling_std_5','rolling_mean_10',
                'rolling_std_10','momentum_5','momentum_10'
            ] if c in ml.columns]
            feature_cols = [c for c in feature_cols if c in ml.columns]

            X      = ml[feature_cols].values
            y_ret  = ml['Return_Gold'].values      # هدف: العائد اليومي %
            prices = ml['Price_Gold'].values        # للعرض فقط
            dates  = ml['Date'].values

            split      = int(len(X) * (1 - test_pct / 100))
            X_tr, X_te = X[:split], X[split:]
            y_tr, y_te = y_ret[:split], y_ret[split:]
            prices_te  = prices[split:]
            dates_te   = dates[split:]

            if model_name == "Random Forest":
                mdl = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            elif model_name == "Linear Regression":
                mdl = LinearRegression()
            elif model_name == "Neural Network (MLP)":
                from sklearn.neural_network import MLPRegressor
                from sklearn.preprocessing import StandardScaler
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
                X_tr_sc  = scaler_X.fit_transform(X_tr)
                X_te_sc  = scaler_X.transform(X_te)
                y_tr_sc  = scaler_y.fit_transform(y_tr.reshape(-1, 1)).ravel()
                mdl = MLPRegressor(
                    hidden_layer_sizes=(128, 64, 32),
                    activation='relu', solver='adam',
                    max_iter=500, random_state=42,
                    early_stopping=True, validation_fraction=0.1,
                    n_iter_no_change=20
                )
                mdl.fit(X_tr_sc, y_tr_sc)
                preds_ret = scaler_y.inverse_transform(
                    mdl.predict(X_te_sc).reshape(-1, 1)).ravel()
            elif model_name == "LightGBM":
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
                try:
                    from xgboost import XGBRegressor
                    mdl = XGBRegressor(n_estimators=300, learning_rate=0.05,
                                       max_depth=6, random_state=42, verbosity=0)
                except ImportError:
                    st.error("XGBoost not available. Choose another model.")
                    st.stop()

            if model_name not in ("Neural Network (MLP)",):
                mdl.fit(X_tr, y_tr)
                preds_ret = mdl.predict(X_te)          # توقع العائد اليومي %

            # ── 1-step-ahead reconstruction ───────────────────────────────────
            # Each day anchored to ACTUAL previous price — no error compounding
            preds_prices = np.zeros(len(prices_te))
            preds_prices[0] = prices_te[0]
            for i in range(1, len(prices_te)):
                preds_prices[i] = prices_te[i - 1] * (1 + preds_ret[i] / 100)

            # ── Metrics ──────────────────────────────────────────────────────
            rmse_ret = np.sqrt(mean_squared_error(y_te, preds_ret))
            mae_ret  = mean_absolute_error(y_te, preds_ret)
            r2_ret   = r2_score(y_te, preds_ret)

            rmse_price = np.sqrt(mean_squared_error(prices_te[1:], preds_prices[1:]))
            mae_price  = mean_absolute_error(prices_te[1:], preds_prices[1:])

            # Directional accuracy: % of days model correctly called up/down
            actual_dir = np.sign(y_te[1:])
            pred_dir   = np.sign(preds_ret[1:])
            dir_acc    = float(np.mean(actual_dir == pred_dir) * 100)

            st.success(f"{model_name} trained — results below.")

            st.markdown("""
            <div style="background:#1c1f23;border:1px solid #2e5f65;border-radius:8px;
                        padding:10px 16px;margin-bottom:12px;font-size:0.82rem;color:#e0f7fa;opacity:0.8;">
            <b>1-step-ahead:</b> each day the model receives <i>yesterday's actual price</i>
            and predicts today. No error compounding.
            <b>Directional accuracy</b> = % of days the model correctly predicted up or down.
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
                _pal_cmp  = {"Random Forest":"#ffc72c","XGBoost":"#2e5f65",
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
                        paper_bgcolor='#1c1f23', plot_bgcolor='#09090b',
                        yaxis_title="Growth ($1)", hovermode='x unified',
                        title="All Models — Cumulative Strategy Return",
                        legend=dict(orientation='h', y=1.08, font=dict(size=10)),
                        margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(_cmp_fig, width='stretch')

            # Trading Signal + Strategy vs Buy & Hold
            st.markdown('---')
            sig_col, strat_col = st.columns([1, 2])
            last_pred_ret = float(preds_ret[-1])
            if last_pred_ret > 0.15:
                sig_txt, sig_color = 'Buy', '#22c55e'
                sig_desc = f'Model expects +{last_pred_ret:.2f}% next session'
            elif last_pred_ret < -0.15:
                sig_txt, sig_color = 'Sell / Avoid', '#ef4444'
                sig_desc = f'Model expects {last_pred_ret:.2f}% next session'
            else:
                sig_txt, sig_color = 'Hold / Neutral', '#ffc72c'
                sig_desc = f'Weak signal ({last_pred_ret:+.2f}%)'
            with sig_col:
                st.markdown(
                    f'<div style="background:#1c1f23;border:1px solid #2e5f65;'
                    f'border-radius:12px;padding:24px;text-align:center;">'
                    f'<div style="font-size:.78rem;color:#e0f7fa;opacity:.6;margin-bottom:6px;">'
                    f'Signal (end of test period)</div>'
                    f'<div style="font-size:2.2rem;font-weight:900;color:{sig_color};">'
                    f'{sig_txt}</div>'
                    f'<div style="font-size:.78rem;color:#e0f7fa;opacity:.5;margin-top:6px;">'
                    f'{sig_desc}</div></div>',
                    unsafe_allow_html=True)
            strat_rets = np.where(preds_ret[:-1] > 0, y_te[1:], 0.0)
            bh_rets    = y_te[1:]
            sharpe_s  = float(np.mean(strat_rets)/np.std(strat_rets)*np.sqrt(252)) \
                        if np.std(strat_rets) > 1e-9 else 0.0
            sharpe_bh = float(np.mean(bh_rets)/np.std(bh_rets)*np.sqrt(252)) \
                        if np.std(bh_rets) > 1e-9 else 0.0
            cum_s  = np.cumprod(1 + strat_rets / 100)
            cum_bh = np.cumprod(1 + bh_rets    / 100)
            with strat_col:
                st.markdown('**Strategy vs Buy & Hold**')
                sh1, sh2 = st.columns(2)
                sh1.metric('Sharpe (Model Strategy)', f'{sharpe_s:.2f}',
                           f'{sharpe_s - sharpe_bh:+.2f} vs B&H')
                sh2.metric('Sharpe (Buy & Hold)', f'{sharpe_bh:.2f}')
                _fig_s = go.Figure()
                _fig_s.add_trace(go.Scatter(x=dates_te[1:], y=cum_bh,
                                            name='Buy & Hold',
                                            line=dict(color='#e0f7fa', width=1.5)))
                _fig_s.add_trace(go.Scatter(x=dates_te[1:], y=cum_s,
                                            name='Model Strategy',
                                            line=dict(color='#ffc72c', width=2)))
                _fig_s.update_layout(
                    height=200, template='plotly_dark',
                    paper_bgcolor='#1c1f23', plot_bgcolor='#09090b',
                    yaxis_title='Growth ($1)', hovermode='x unified',
                    legend=dict(orientation='h', y=1.02, font=dict(size=10)),
                    margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(_fig_s, width='stretch')
            st.markdown('---')

            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=dates_te, y=prices_te, name='Actual Price',
                                           line=dict(color='#e0f7fa', width=2)))
            fig_pred.add_trace(go.Scatter(x=dates_te, y=preds_prices,
                                           name=f'Predicted ({model_name})',
                                           line=dict(color='#ffc72c', width=2, dash='dash')))
            fig_pred.update_layout(height=420, template='plotly_dark',
                                   paper_bgcolor='#1c1f23', plot_bgcolor='#09090b',
                                   title=f"{model_name}: Reconstructed Price Path",
                                   yaxis_title="Price (USD)",
                                   hovermode='x unified',
                                   legend=dict(orientation='h', y=1.02),
                                   margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_pred, width='stretch')

            # ── مخطط العوائد اليومية ──────────────────────────────────────────
            with st.expander("Daily Returns: Actual vs Predicted"):
                fig_ret = go.Figure()
                fig_ret.add_trace(go.Scatter(x=dates_te, y=y_te, name='Actual Return',
                                              line=dict(color='#e0f7fa', width=1.2)))
                fig_ret.add_trace(go.Scatter(x=dates_te, y=preds_ret,
                                              name='Predicted Return',
                                              line=dict(color='#ffc72c', width=1.2, dash='dash')))
                fig_ret.add_hline(y=0, line=dict(color='#4a5568', width=1, dash='dot'))
                fig_ret.update_layout(height=280, template='plotly_dark',
                                      paper_bgcolor='#1c1f23', plot_bgcolor='#09090b',
                                      yaxis_title="Daily Return (%)",
                                      hovermode='x unified',
                                      legend=dict(orientation='h', y=1.02),
                                      margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_ret, width='stretch')

            if model_name in ["Random Forest", "XGBoost"] and hasattr(mdl, "feature_importances_"):
                st.subheader("Feature Importance")
                fi     = pd.Series(mdl.feature_importances_,
                                   index=feature_cols).sort_values(ascending=True)
                fig_fi = go.Figure(go.Bar(x=fi.values, y=fi.index,
                                           orientation='h', marker_color='#ffc72c'))
                fig_fi.update_layout(height=max(250, len(feature_cols) * 30),
                                     template='plotly_dark',
                                     paper_bgcolor='#1c1f23', plot_bgcolor='#09090b',
                                     xaxis_title="Importance",
                                     margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_fi, width='stretch')

            with st.expander("Residual Analysis"):
                residuals = y_te - preds_ret
                fig_res   = px.histogram(residuals, nbins=60,
                                          color_discrete_sequence=['#2e5f65'],
                                          labels={'value': 'Residual (% return)'})
                fig_res.update_layout(height=250, template='plotly_dark',
                                      paper_bgcolor='#1c1f23', plot_bgcolor='#09090b',
                                      xaxis_title="Residual (% return)", showlegend=False)
                st.plotly_chart(fig_res, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Forecast":

    st.markdown("<h1>30-Day Gold Price Forecast</h1>", unsafe_allow_html=True)
    st.markdown("Recursive multi-step forecast trained on all available historical data.")

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        fc_model = st.selectbox("Model",
                                 ["Random Forest", "XGBoost", "LightGBM", "Prophet", "Linear Regression"],
                                 key="fc_model")
    with fc2:
        fc_lags = st.slider("Lag Features (days)", 5, 60, 20, key="fc_lags")
    with fc3:
        fc_days = st.slider("Forecast Horizon (days)", 7, 90, 30, key="fc_days")

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

            for lag in range(1, fc_lags + 1):
                fc_df[f'lag_{lag}'] = fc_df['Return_Gold'].shift(lag)

            # Rolling statistics & momentum (time-series features)
            for _w in [5, 10, 20]:
                fc_df[f'rolling_mean_{_w}'] = fc_df['Return_Gold'].rolling(_w).mean()
                fc_df[f'rolling_std_{_w}']  = fc_df['Return_Gold'].rolling(_w).std()
            fc_df['momentum_5']  = fc_df['Price_Gold'].pct_change(5)  * 100
            fc_df['momentum_10'] = fc_df['Price_Gold'].pct_change(10) * 100
            fc_df = fc_df.dropna().reset_index(drop=True)

            ts_feat_cols = [c for c in [
                'rolling_mean_5','rolling_std_5','rolling_mean_10',
                'rolling_std_10','momentum_5','momentum_10'
            ] if c in fc_df.columns]
            feature_cols = [f'lag_{i}' for i in range(1, fc_lags + 1)] + ret_extra_cols + ts_feat_cols
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
            if fc_model != "Prophet":
                horizon_factor = np.array([1 + 0.05 * i for i in range(fc_days)])
                upper_band     = forecast_prices + cv_rmse * horizon_factor
                lower_band     = forecast_prices - cv_rmse * horizon_factor
            upper_band = np.array(upper_band)
            lower_band = np.array(lower_band)

            last_actual   = float(fc_df['Price_Gold'].iloc[-1])
            last_forecast = float(forecast_prices[-1])
            change_usd    = last_forecast - last_actual
            change_pct    = (change_usd / last_actual) * 100

            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("Last Actual Price",       f"${last_actual:,.2f}")
            sm2.metric(f"Day {fc_days} Forecast", f"${last_forecast:,.2f}",
                       f"{change_usd:+,.2f} ({change_pct:+.1f}%)")
            sm3.metric("Forecast Peak",   f"${forecast_prices.max():,.2f}")
            sm4.metric("Forecast Trough", f"${forecast_prices.min():,.2f}")

            st.markdown("---")

            hist   = fc_df[['Date', 'Price_Gold']].tail(180)
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                x=list(forecast_dates) + list(forecast_dates[::-1]),
                y=list(upper_band) + list(lower_band[::-1]),
                fill='toself', fillcolor='rgba(255,199,44,0.10)',
                line=dict(color='rgba(255,199,44,0)'),
                hoverinfo='skip', showlegend=True, name='Confidence Band'))
            fig_fc.add_trace(go.Scatter(x=forecast_dates, y=upper_band,
                                         line=dict(color='rgba(255,199,44,0.4)',
                                                   width=1, dash='dot'),
                                         name='Upper Bound',
                                         hovertemplate='$%{y:,.2f}'))
            fig_fc.add_trace(go.Scatter(x=forecast_dates, y=lower_band,
                                         line=dict(color='rgba(255,199,44,0.4)',
                                                   width=1, dash='dot'),
                                         name='Lower Bound',
                                         hovertemplate='$%{y:,.2f}'))
            fig_fc.add_trace(go.Scatter(x=hist['Date'], y=hist['Price_Gold'],
                                         line=dict(color='#e0f7fa', width=2),
                                         name='Historical Price'))
            fig_fc.add_trace(go.Scatter(x=forecast_dates, y=forecast_prices,
                                         line=dict(color='#ffc72c', width=2.5, dash='dash'),
                                         name='Forecast',
                                         hovertemplate='%{x|%b %d, %Y}<br>$%{y:,.2f}<extra></extra>'))
            _vline_x = str(fc_df['Date'].iloc[-1].date())
            fig_fc.add_shape(type="line",
                              x0=_vline_x, x1=_vline_x, y0=0, y1=1,
                              xref="x", yref="paper",
                              line=dict(color='#2e5f65', width=1.5, dash='dot'))
            fig_fc.add_annotation(x=_vline_x, y=0.98,
                                   xref="x", yref="paper",
                                   text="Forecast Start", showarrow=False,
                                   font=dict(color='#2e5f65', size=11),
                                   xanchor="left")
            fig_fc.update_layout(height=500, template='plotly_dark',
                                  paper_bgcolor='#1c1f23', plot_bgcolor='#09090b',
                                  yaxis_title="Price (USD)", hovermode='x unified',
                                  legend=dict(orientation='h', y=1.02, font=dict(size=10)),
                                  margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_fc, width='stretch')

            st.subheader("Daily Forecast Table")
            fc_table = pd.DataFrame({
                'Date':            [d.strftime('%A, %b %d %Y') for d in forecast_dates],
                'Forecast Price':  [f"${p:,.2f}" for p in forecast_prices],
                'Upper Bound':     [f"${u:,.2f}" for u in upper_band],
                'Lower Bound':     [f"${lo:,.2f}" for lo in lower_band],
                'Change vs Today': [f"{((p - last_actual) / last_actual * 100):+.2f}%"
                                    for p in forecast_prices],
            })
            fc_table.index      = range(1, fc_days + 1)
            fc_table.index.name = "Day"
            st.dataframe(fc_table, width='stretch')
            st.download_button(
                label="Download Forecast as CSV",
                data=fc_table.to_csv().encode('utf-8'),
                file_name=f"gold_forecast_{fc_days}days_{fc_model.replace(' ','_')}.csv",
                mime="text/csv",
                use_container_width=True)

            with st.expander("Model Details"):
                rmse_str = f"${cv_rmse:,.2f}"
                st.markdown(f"""
- **Model:** {fc_model}
- **Training rows:** {len(X_all):,}
- **Lag features:** {fc_lags} days
- **CV RMSE (base uncertainty):** {rmse_str}
- **Method:** Recursive — each predicted day feeds the next step
- **Confidence band:** grows by 5% per step to reflect compounding uncertainty
                """)
    else:
        st.info("Configure the settings above and click **Generate Forecast** to run.")
        st.markdown("""
        <div style="background:#1c1f23; border:1px solid #2e5f65; border-radius:12px; padding:20px; margin-top:12px;">
            <h3 style="color:#ffc72c; margin-top:0;">How It Works</h3>
            <ol style="color:#e0f7fa; line-height:2; padding-left:20px;">
                <li>Train the model on <b>all available historical data</b></li>
                <li>Use the last N actual prices as lag input features</li>
                <li>Predict Day 1 — feed that prediction into Day 2, and so on</li>
                <li>Confidence band widens over the horizon to reflect growing uncertainty</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "About":

    st.markdown("<h1>About This App</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#1c1f23; border:1px solid #2e5f65; border-radius:12px; padding:24px 28px; margin-bottom:20px;">
        <h3 style="color:#ffc72c; margin-top:0;">Gold Price Prediction App</h3>
        <p style="color:#e0f7fa; line-height:1.8;">
            An interactive dashboard and machine learning prediction tool for gold price analysis,
            using historical data from Yahoo Finance covering gold futures, crude oil,
            the US Dollar Index, and the S&P 500.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background:#1c1f23; border:1px solid #2e5f65; border-radius:12px; padding:20px; margin-bottom:16px;">
            <h3 style="color:#ffc72c; margin-top:0;">Data</h3>
            <ul style="color:#e0f7fa; line-height:2; margin:0; padding-left:20px;">
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
        <div style="background:#1c1f23; border:1px solid #2e5f65; border-radius:12px; padding:20px;">
            <h3 style="color:#ffc72c; margin-top:0;">Dashboard Features</h3>
            <ul style="color:#e0f7fa; line-height:2; margin:0; padding-left:20px;">
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
        <div style="background:#1c1f23; border:1px solid #2e5f65; border-radius:12px; padding:20px; margin-bottom:16px;">
            <h3 style="color:#ffc72c; margin-top:0;">ML Models (Prediction)</h3>
            <ul style="color:#e0f7fa; line-height:2; margin:0; padding-left:20px;">
                <li><b style="color:#ffc72c;">Random Forest</b> — ensemble of decision trees</li>
                <li><b style="color:#ffc72c;">XGBoost</b> — extreme gradient boosting</li>
                <li><b style="color:#ffc72c;">LightGBM</b> — fast gradient boosting, less overfitting</li>
                <li><b style="color:#ffc72c;">Neural Network (MLP)</b> — multi-layer perceptron</li>
                <li><b style="color:#ffc72c;">Linear Regression</b> — baseline model</li>
            </ul>
        </div>
        <div style="background:#1c1f23; border:1px solid #2e5f65; border-radius:12px; padding:20px;">
            <h3 style="color:#ffc72c; margin-top:0;">ML Models (Forecast)</h3>
            <ul style="color:#e0f7fa; line-height:2; margin:0; padding-left:20px;">
                <li><b style="color:#ffc72c;">Prophet</b> — Meta's time-series model, handles trend &amp; seasonality</li>
                <li><b style="color:#ffc72c;">LightGBM / XGBoost / RF</b> — recursive lag-feature forecast</li>
                <li><b style="color:#ffc72c;">Linear Regression</b> — baseline</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#1c1f23; border:1px solid #2e5f65; border-radius:12px; padding:20px;">
            <h3 style="color:#ffc72c; margin-top:0;">Tech Stack</h3>
            <ul style="color:#e0f7fa; line-height:2; margin:0; padding-left:20px;">
                <li>Python 3.11 + Streamlit</li>
                <li>Plotly (interactive charts)</li>
                <li>scikit-learn / XGBoost / LightGBM / Prophet</li>
                <li>pandas / numpy</li>
                <li>GitHub Actions (auto daily update)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#1c1f23; border:1px solid #2e5f65; border-radius:12px; padding:20px; margin-top:4px;">
        <p style="color:#e0f7fa; margin:0; font-size:0.85rem; opacity:0.7; text-align:center;">
            Built with Streamlit · Data from Yahoo Finance · Auto-updated daily
        </p>
    </div>
    """, unsafe_allow_html=True)
