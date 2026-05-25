import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page Config
st.set_page_config(
    page_title="Gold Price Prediction",
    page_icon="gold",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
/* Background */
.stApp { background-color: #09090b; }
[data-testid="stSidebar"] { background-color: #0d0d10; border-right: 1px solid #2e5f65; }

/* Headings */
h1 { color: #ffc72c !important; letter-spacing: 1px; }
h2, h3 { color: #ffc72c !important; }

/* Metric cards */
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

/* Buttons */
.stButton > button {
    background: #ffc72c;
    color: #09090b;
    font-weight: 800;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    transition: all 0.2s ease;
    letter-spacing: 0.5px;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); box-shadow: 0 4px 15px rgba(255,199,44,0.35); }

/* Selectbox / Sliders */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stMultiSelect"] label { color: #e0f7fa !important; font-size: 0.9rem; opacity: 0.8; }

/* Divider */
hr { border-color: #2e5f65; }

/* Expander */
[data-testid="stExpander"] { border: 1px solid #2e5f65; border-radius: 8px; background: #1c1f23; }

/* Sidebar text */
.stRadio label { color: #e0f7fa !important; }
.block-container { padding-top: 1.5rem; }

/* Info/warning boxes */
[data-testid="stAlert"] { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# Data Loading & Cleaning
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

    # Gold price columns
    for col in ['Price_Gold', 'High_Gold', 'Low_Gold', 'Open_Gold']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].apply(clean_currency), errors='coerce')

    if 'Change%_Gold' in df.columns:
        df['Change%_Gold'] = df['Change%_Gold'].apply(clean_percent)

    if 'Volume_Gold' in df.columns:
        df['Volume_Gold'] = df['Volume_Gold'].apply(clean_volume)
        df['Volume_Gold'] = pd.to_numeric(df['Volume_Gold'], errors='coerce')

    # Other assets
    for col in ['Price_Oil', 'Price_Dollar', 'High_Dollar', 'Low_Dollar',
                'Open_Dollar', 'Volume_Dollar', 'Price_Stocks', 'High_Stocks',
                'Low_Stocks', 'Open_Stocks', 'Volume_Stocks']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    return df


df = load_data()

# Sidebar
with st.sidebar:
    # Logo / Header
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 8px 0;">
        <div style="font-size:1.25rem; font-weight:800; color:#ffc72c; letter-spacing:1px; margin-top:6px;">
            Gold Price
        </div>
        <div style="font-size:0.78rem; color:#e0f7fa; opacity:0.7; margin-top:2px; letter-spacing:2px;">
            PREDICTION APP
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#2e5f65; margin:8px 0 12px 0;'>", unsafe_allow_html=True)

    # Navigation
    page = st.radio(
        "nav",
        ["Dashboard", "Prediction", "About"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#2e5f65; margin:12px 0;'>", unsafe_allow_html=True)

    # Date range (only show for Dashboard & Prediction)
    if page != "About":
        st.markdown("**Date Range**")
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()

        start_date = st.date_input("From", value=pd.to_datetime("2010-01-01").date(),
                                   min_value=min_date, max_value=max_date)
        end_date = st.date_input("To", value=max_date,
                                 min_value=min_date, max_value=max_date)

        st.markdown("<hr style='border-color:#2e5f65; margin:12px 0;'>", unsafe_allow_html=True)
    else:
        start_date = pd.to_datetime("2010-01-01").date()
        end_date = df['Date'].max().date()

    # Stats
    total_rows = len(df)
    st.markdown(f"""
    <div style="font-size:0.78rem; color:#2e5f65; line-height:2;">
        &nbsp;Source: Yahoo Finance<br>
        &nbsp;Period: 1986 – present<br>
        &nbsp;{total_rows:,} trading days
    </div>
    """, unsafe_allow_html=True)

# Filter
mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
filtered_df = df[mask].copy()


# ==============================================================================
# DASHBOARD
# ==============================================================================
if page == "Dashboard":

    st.markdown("<h1>Gold Price Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(f"**{start_date}** → **{end_date}** &nbsp;|&nbsp; {len(filtered_df):,} trading days")

    # KPI Cards
    gold_clean = filtered_df.dropna(subset=['Price_Gold'])
    if not gold_clean.empty:
        latest = gold_clean.iloc[-1]
        first = gold_clean.iloc[0]
        delta = latest['Price_Gold'] - first['Price_Gold']
        pct = (delta / first['Price_Gold']) * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price", f"${latest['Price_Gold']:,.2f}", f"{delta:+,.2f} ({pct:+.1f}%)")
        c2.metric("Period High", f"${gold_clean['High_Gold'].max():,.2f}")
        c3.metric("Period Low", f"${gold_clean['Low_Gold'].min():,.2f}")
        c4.metric("Average Price", f"${gold_clean['Price_Gold'].mean():,.2f}")

    st.markdown("---")

    # Gold Price Chart
    chart_col, ctrl_col = st.columns([4, 1])
    with chart_col:
        st.subheader("Gold Price Over Time")
    with ctrl_col:
        st.markdown("<div style='padding-top:28px'></div>", unsafe_allow_html=True)
        chart_type = st.radio("Chart type", ["Candlestick", "Line"],
                              horizontal=True, label_visibility="collapsed")

    candle_df = filtered_df.dropna(subset=['Open_Gold', 'High_Gold', 'Low_Gold', 'Price_Gold'])

    fig = go.Figure()

    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=candle_df['Date'],
            open=candle_df['Open_Gold'],
            high=candle_df['High_Gold'],
            low=candle_df['Low_Gold'],
            close=candle_df['Price_Gold'],
            name='OHLC',
            increasing=dict(line=dict(color='#ffc72c', width=1),
                            fillcolor='rgba(255,199,44,0.85)'),
            decreasing=dict(line=dict(color='#ef4444', width=1),
                            fillcolor='rgba(239,68,68,0.75)'),
        ))
        # Volume bars underneath
        if 'Volume_Gold' in candle_df.columns:
            vol = candle_df.dropna(subset=['Volume_Gold'])
            colors_v = ['rgba(255,199,44,0.4)' if c >= o else 'rgba(239,68,68,0.35)'
                        for c, o in zip(vol['Price_Gold'], vol['Open_Gold'])]
            fig.add_trace(go.Bar(
                x=vol['Date'], y=vol['Volume_Gold'],
                name='Volume', marker_color=colors_v,
                yaxis='y2', showlegend=False
            ))
        fig.update_layout(
            yaxis2=dict(overlaying='y', side='right', showgrid=False,
                        showticklabels=False,
                        range=[0, candle_df['Volume_Gold'].max() * 5]
                        if 'Volume_Gold' in candle_df.columns else {}),
            xaxis_rangeslider_visible=False,
        )
    else:
        fig.add_trace(go.Scatter(
            x=filtered_df['Date'], y=filtered_df['Price_Gold'],
            mode='lines', name='Close',
            line=dict(color='#ffc72c', width=2),
            fill='tozeroy', fillcolor='rgba(255,199,44,0.08)'
        ))
        fig.add_trace(go.Scatter(
            x=filtered_df['Date'], y=filtered_df['High_Gold'],
            mode='lines', name='High',
            line=dict(color='rgba(0,200,0,0.4)', width=1, dash='dot')
        ))
        fig.add_trace(go.Scatter(
            x=filtered_df['Date'], y=filtered_df['Low_Gold'],
            mode='lines', name='Low',
            line=dict(color='rgba(255,80,80,0.4)', width=1, dash='dot'),
            fill='tonexty', fillcolor='rgba(180,180,180,0.04)'
        ))

    fig.update_layout(
        height=480, template='plotly_dark', paper_bgcolor='#1c1f23', plot_bgcolor='#09090b',
        xaxis_title="Date", yaxis_title="Price (USD)",
        hovermode='x unified',
        legend=dict(orientation='h', y=1.02, x=0),
        margin=dict(l=0, r=0, t=10, b=0)
    )
    st.plotly_chart(fig, width='stretch')

    # ── Technical Indicators ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Technical Indicators")

    ti_c1, ti_c2 = st.columns(2)
    with ti_c1:
        bb_period = st.slider("Bollinger Bands Period", 10, 50, 20, key="bb_p")
    with ti_c2:
        rsi_period = st.slider("RSI Period", 7, 21, 14, key="rsi_p")

    tech_df = filtered_df[['Date', 'Price_Gold']].dropna().copy().reset_index(drop=True)

    # Bollinger Bands
    tech_df['SMA']      = tech_df['Price_Gold'].rolling(bb_period).mean()
    tech_df['STD']      = tech_df['Price_Gold'].rolling(bb_period).std()
    tech_df['BB_upper'] = tech_df['SMA'] + 2 * tech_df['STD']
    tech_df['BB_lower'] = tech_df['SMA'] - 2 * tech_df['STD']
    tech_df['BB_pct']   = (tech_df['Price_Gold'] - tech_df['BB_lower']) / (
                           tech_df['BB_upper'] - tech_df['BB_lower'])

    # RSI
    delta = tech_df['Price_Gold'].diff()
    gain  = delta.clip(lower=0).rolling(rsi_period).mean()
    loss  = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs    = gain / loss.replace(0, np.nan)
    tech_df['RSI'] = 100 - (100 / (1 + rs))

    rsi_valid = tech_df['RSI'].dropna()
    current_rsi = rsi_valid.iloc[-1] if not rsi_valid.empty else None
    latest_price = tech_df['Price_Gold'].iloc[-1]
    latest_upper = tech_df['BB_upper'].iloc[-1]
    latest_lower = tech_df['BB_lower'].iloc[-1]
    latest_sma   = tech_df['SMA'].iloc[-1]

    # Signal cards
    if current_rsi is not None:
        if current_rsi >= 70:
            rsi_label, rsi_delta = "Overbought", f"{current_rsi:.1f} — Sell signal"
        elif current_rsi <= 30:
            rsi_label, rsi_delta = "Oversold", f"{current_rsi:.1f} — Buy signal"
        else:
            rsi_label, rsi_delta = "Neutral", f"{current_rsi:.1f}"

        if not np.isnan(latest_upper):
            bb_width = latest_upper - latest_lower
            bb_pos   = (latest_price - latest_lower) / bb_width * 100 if bb_width else 50
            if bb_pos >= 80:
                bb_label = "Near Upper Band"
            elif bb_pos <= 20:
                bb_label = "Near Lower Band"
            else:
                bb_label = "Inside Bands"
        else:
            bb_pos, bb_label = 50, "—"

        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("RSI", f"{current_rsi:.1f}", rsi_label)
        sm2.metric("SMA (BB mid)", f"${latest_sma:,.2f}" if not np.isnan(latest_sma) else "—")
        sm3.metric("Upper Band", f"${latest_upper:,.2f}" if not np.isnan(latest_upper) else "—")
        sm4.metric("Lower Band", f"${latest_lower:,.2f}" if not np.isnan(latest_lower) else "—")

    bb_col, rsi_col = st.columns(2)

    # ── Bollinger Bands Chart ─────────────────────────────────────────────────
    with bb_col:
        st.markdown("**Bollinger Bands**")
        bb_plot = tech_df.dropna(subset=['SMA'])
        fig_bb = go.Figure()

        # Shaded band area
        fig_bb.add_trace(go.Scatter(
            x=bb_plot['Date'], y=bb_plot['BB_upper'],
            line=dict(color='rgba(46,95,101,0)', width=0),
            showlegend=False, name='Upper'
        ))
        fig_bb.add_trace(go.Scatter(
            x=bb_plot['Date'], y=bb_plot['BB_lower'],
            fill='tonexty', fillcolor='rgba(46,95,101,0.18)',
            line=dict(color='rgba(46,95,101,0)', width=0),
            showlegend=False, name='Lower'
        ))
        # Bands
        fig_bb.add_trace(go.Scatter(
            x=bb_plot['Date'], y=bb_plot['BB_upper'],
            line=dict(color='#2e5f65', width=1.2, dash='dot'),
            name='Upper Band'
        ))
        fig_bb.add_trace(go.Scatter(
            x=bb_plot['Date'], y=bb_plot['BB_lower'],
            line=dict(color='#2e5f65', width=1.2, dash='dot'),
            name='Lower Band'
        ))
        # SMA
        fig_bb.add_trace(go.Scatter(
            x=bb_plot['Date'], y=bb_plot['SMA'],
            line=dict(color='#e0f7fa', width=1.5, dash='dash'),
            name=f'SMA {bb_period}'
        ))
        # Price
        fig_bb.add_trace(go.Scatter(
            x=bb_plot['Date'], y=bb_plot['Price_Gold'],
            line=dict(color='#ffc72c', width=2),
            name='Gold Price'
        ))
        fig_bb.update_layout(
            height=360, template='plotly_dark',
            paper_bgcolor='#1c1f23', plot_bgcolor='#09090b',
            yaxis_title="Price (USD)", hovermode='x unified',
            legend=dict(orientation='h', y=1.02, font=dict(size=10)),
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig_bb, width='stretch')

    # ── RSI Chart ─────────────────────────────────────────────────────────────
    with rsi_col:
        st.markdown("**Relative Strength Index (RSI)**")
        rsi_plot = tech_df.dropna(subset=['RSI'])
        fig_rsi = go.Figure()

        # Overbought / Oversold zones
        fig_rsi.add_hrect(y0=70, y1=100, fillcolor='rgba(239,68,68,0.08)',
                          line_width=0, annotation_text="Overbought",
                          annotation_position="top right",
                          annotation=dict(font_size=10, font_color='#ef4444'))
        fig_rsi.add_hrect(y0=0, y1=30, fillcolor='rgba(34,197,94,0.08)',
                          line_width=0, annotation_text="Oversold",
                          annotation_position="bottom right",
                          annotation=dict(font_size=10, font_color='#22c55e'))

        # Reference lines
        fig_rsi.add_hline(y=70, line=dict(color='#ef4444', width=1, dash='dot'))
        fig_rsi.add_hline(y=30, line=dict(color='#22c55e', width=1, dash='dot'))
        fig_rsi.add_hline(y=50, line=dict(color='#4a5568', width=1, dash='dot'))

        # RSI line — colored by zone
        rsi_colors = ['#ef4444' if v >= 70 else '#22c55e' if v <= 30 else '#ffc72c'
                      for v in rsi_plot['RSI']]
        fig_rsi.add_trace(go.Scatter(
            x=rsi_plot['Date'], y=rsi_plot['RSI'],
            mode='lines', name=f'RSI ({rsi_period})',
            line=dict(color='#ffc72c', width=2)
        ))

        fig_rsi.update_layout(
            height=360, template='plotly_dark',
            paper_bgcolor='#1c1f23', plot_bgcolor='#09090b',
            yaxis=dict(title="RSI", range=[0, 100],
                       tickvals=[0, 30, 50, 70, 100]),
            hovermode='x unified',
            legend=dict(orientation='h', y=1.02, font=dict(size=10)),
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig_rsi, width='stretch')

    st.markdown("---")

    # Comparison & Correlation
    left, right = st.columns(2)

    with left:
        st.subheader("Asset Comparison (Normalized)")
        cmp = filtered_df[['Date', 'Price_Gold', 'Price_Oil',
                            'Price_Dollar', 'Price_Stocks']].dropna()
        if len(cmp) > 1:
            vals = MinMaxScaler((0, 100)).fit_transform(
                cmp[['Price_Gold', 'Price_Oil', 'Price_Dollar', 'Price_Stocks']]
            )
            norm = pd.DataFrame(vals, columns=['Gold', 'Oil', 'Dollar Index', 'S&P 500'])
            norm['Date'] = cmp['Date'].values

            fig2 = go.Figure()
            palette = {'Gold': '#ffc72c', 'Oil': '#2e5f65',
                       'Dollar Index': '#e0f7fa', 'S&P 500': '#4a5568'}
            for col, color in palette.items():
                fig2.add_trace(go.Scatter(x=norm['Date'], y=norm[col],
                                          mode='lines', name=col,
                                          line=dict(color=color, width=1.5)))
            fig2.update_layout(height=340, template='plotly_dark', paper_bgcolor='#1c1f23', plot_bgcolor='#09090b',
                               yaxis_title="Normalized (0–100)",
                               hovermode='x unified',
                               legend=dict(orientation='h', y=1.02))
            st.plotly_chart(fig2, width='stretch')
        else:
            st.info("Not enough data for comparison.")

    with right:
        st.subheader("Correlation Heatmap")
        corr_df = filtered_df[['Price_Gold', 'Price_Oil',
                                'Price_Dollar', 'Price_Stocks']].dropna()
        if len(corr_df) > 1:
            fig3 = px.imshow(
                corr_df.corr(),
                text_auto='.2f',
                color_continuous_scale='RdYlGn',
                zmin=-1, zmax=1
            )
            fig3.update_layout(height=340, template='plotly_dark', paper_bgcolor='#1c1f23', plot_bgcolor='#09090b')
            st.plotly_chart(fig3, width='stretch')
        else:
            st.info("Not enough data for correlation.")

    # Yearly Average & Volume
    l2, r2 = st.columns(2)

    with l2:
        st.subheader("Yearly Average Gold Price")
        yearly = filtered_df.groupby('Year')['Price_Gold'].mean().reset_index()
        fig4 = px.bar(yearly, x='Year', y='Price_Gold',
                      color='Price_Gold', color_continuous_scale=[[0, '#2e5f65'], [0.5, '#ffc72c'], [1, '#ffffff']],
                      labels={'Price_Gold': 'Avg Price (USD)'})
        fig4.update_layout(height=300, template='plotly_dark', paper_bgcolor='#1c1f23', plot_bgcolor='#09090b',
                           showlegend=False, xaxis_title="")
        st.plotly_chart(fig4, width='stretch')

    with r2:
        st.subheader("Trading Volume")
        vol = filtered_df[['Date', 'Volume_Gold']].dropna()
        if not vol.empty:
            fig5 = go.Figure(go.Bar(
                x=vol['Date'], y=vol['Volume_Gold'],
                marker_color='rgba(255,199,44,0.7)'
            ))
            fig5.update_layout(height=300, template='plotly_dark', paper_bgcolor='#1c1f23', plot_bgcolor='#09090b',
                               yaxis_title="Volume", xaxis_title="")
            st.plotly_chart(fig5, width='stretch')
        else:
            st.info("No volume data in selected range.")

    # Daily Change Distribution
    st.subheader("Daily Change % Distribution")
    chg = filtered_df['Change%_Gold'].dropna() * 100
    if not chg.empty:
        fig6 = px.histogram(chg, nbins=80, color_discrete_sequence=['#ffc72c'],
                            labels={'value': 'Daily Change (%)', 'count': 'Days'})
        fig6.update_layout(height=250, template='plotly_dark', paper_bgcolor='#1c1f23', plot_bgcolor='#09090b',
                           xaxis_title="Daily Change (%)", showlegend=False)
        st.plotly_chart(fig6, width='stretch')

    # Raw Data
    with st.expander("View Raw Data"):
        st.dataframe(
            filtered_df[['Date', 'Price_Gold', 'High_Gold', 'Low_Gold',
                         'Price_Oil', 'Price_Dollar', 'Price_Stocks']]
            .dropna().set_index('Date'),
            width='stretch'
        )


# ==============================================================================
# PREDICTION
# ==============================================================================
elif page == "Prediction":

    st.markdown("<h1>Gold Price Prediction</h1>", unsafe_allow_html=True)
    st.markdown("Train a machine learning model to predict gold prices from historical patterns.")

    c1, c2, c3 = st.columns(3)
    with c1:
        model_name = st.selectbox("Model", ["Random Forest", "Linear Regression", "XGBoost", "LSTM (Deep Learning)"])
    with c2:
        test_pct = st.slider("Test Size %", 10, 40, 20)
    with c3:
        n_lags = st.slider("Lag Features (days)", 1, 30, 5)

    extra_feats = st.multiselect(
        "Additional Features",
        ['Price_Oil', 'Price_Dollar', 'Price_Stocks'],
        default=['Price_Oil', 'Price_Dollar']
    )

    run = st.button("Train Model", type="primary", width='stretch')

    if run:
        with st.spinner("Training ... please wait"):

            cols = ['Date', 'Price_Gold'] + extra_feats
            ml = filtered_df[cols].dropna().sort_values('Date').reset_index(drop=True)

            if model_name == "LSTM (Deep Learning)":
                try:
                    import tensorflow as tf
                    from tensorflow.keras.models import Sequential
                    from tensorflow.keras.layers import LSTM, Dense, Dropout

                    scaler = MinMaxScaler()
                    scaled = scaler.fit_transform(ml[['Price_Gold']])

                    SEQ_LEN = max(n_lags, 10)
                    X_seq, y_seq = [], []
                    for i in range(SEQ_LEN, len(scaled)):
                        X_seq.append(scaled[i - SEQ_LEN:i, 0])
                        y_seq.append(scaled[i, 0])
                    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
                    X_seq = X_seq.reshape(X_seq.shape[0], X_seq.shape[1], 1)

                    split = int(len(X_seq) * (1 - test_pct / 100))
                    X_tr, X_te = X_seq[:split], X_seq[split:]
                    y_tr, y_te = y_seq[:split], y_seq[split:]

                    model = Sequential([
                        LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, 1)),
                        Dropout(0.2),
                        LSTM(32),
                        Dropout(0.2),
                        Dense(1)
                    ])
                    model.compile(optimizer='adam', loss='mse')
                    model.fit(X_tr, y_tr, epochs=20, batch_size=32, verbose=0)

                    preds_scaled = model.predict(X_te, verbose=0)
                    preds = scaler.inverse_transform(preds_scaled).flatten()
                    actual = scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()
                    test_dates = ml['Date'].iloc[split + SEQ_LEN:].reset_index(drop=True)

                    rmse = np.sqrt(mean_squared_error(actual, preds))
                    mae = mean_absolute_error(actual, preds)
                    r2 = r2_score(actual, preds)

                    st.success("LSTM model trained successfully!")
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("RMSE", f"${rmse:,.2f}")
                    mc2.metric("MAE", f"${mae:,.2f}")
                    mc3.metric("R2 Score", f"{r2:.4f}")

                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Scatter(x=test_dates, y=actual, name='Actual',
                                                   line=dict(color='#e0f7fa', width=2)))
                    fig_pred.add_trace(go.Scatter(x=test_dates, y=preds, name='Predicted (LSTM)',
                                                   line=dict(color='#ffc72c', width=2, dash='dash')))
                    fig_pred.update_layout(
                        height=420, template='plotly_dark', paper_bgcolor='#1c1f23', plot_bgcolor='#09090b',
                        title="LSTM: Actual vs Predicted", hovermode='x unified',
                        legend=dict(orientation='h', y=1.02),
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    st.plotly_chart(fig_pred, width='stretch')

                except ImportError:
                    st.error(
                        "TensorFlow is not available in this environment. "
                        "LSTM requires TensorFlow which is not supported on Streamlit Cloud. "
                        "Please select Random Forest, Linear Regression, or XGBoost instead."
                    )

            else:
                for lag in range(1, n_lags + 1):
                    ml[f'lag_{lag}'] = ml['Price_Gold'].shift(lag)

                ml = ml.dropna().reset_index(drop=True)

                feature_cols = [f'lag_{i}' for i in range(1, n_lags + 1)] + extra_feats
                feature_cols = [c for c in feature_cols if c in ml.columns]

                X = ml[feature_cols].values
                y = ml['Price_Gold'].values
                dates = ml['Date'].values

                split = int(len(X) * (1 - test_pct / 100))
                X_tr, X_te = X[:split], X[split:]
                y_tr, y_te = y[:split], y[split:]
                dates_te = dates[split:]

                if model_name == "Random Forest":
                    mdl = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
                elif model_name == "Linear Regression":
                    mdl = LinearRegression()
                else:
                    try:
                        from xgboost import XGBRegressor
                        mdl = XGBRegressor(n_estimators=300, learning_rate=0.05,
                                           max_depth=6, random_state=42, verbosity=0)
                    except ImportError:
                        st.error("XGBoost is not installed. Please choose another model.")
                        st.stop()

                mdl.fit(X_tr, y_tr)
                preds = mdl.predict(X_te)

                rmse = np.sqrt(mean_squared_error(y_te, preds))
                mae = mean_absolute_error(y_te, preds)
                r2 = r2_score(y_te, preds)

                st.success(f"{model_name} trained successfully!")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("RMSE", f"${rmse:,.2f}")
                mc2.metric("MAE", f"${mae:,.2f}")
                mc3.metric("R2 Score", f"{r2:.4f}")

                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(
                    x=dates_te, y=y_te, name='Actual',
                    line=dict(color='#e0f7fa', width=2)
                ))
                fig_pred.add_trace(go.Scatter(
                    x=dates_te, y=preds, name=f'Predicted ({model_name})',
                    line=dict(color='#ffc72c', width=2, dash='dash')
                ))
                fig_pred.update_layout(
                    height=420, template='plotly_dark', paper_bgcolor='#1c1f23', plot_bgcolor='#09090b',
                    title=f"{model_name}: Actual vs Predicted",
                    hovermode='x unified',
                    legend=dict(orientation='h', y=1.02),
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                st.plotly_chart(fig_pred, width='stretch')

                if model_name in ["Random Forest", "XGBoost"]:
                    st.subheader("Feature Importance")
                    fi = pd.Series(mdl.feature_importances_, index=feature_cols).sort_values(ascending=True)
                    fig_fi = go.Figure(go.Bar(
                        x=fi.values, y=fi.index, orientation='h',
                        marker_color='#ffc72c'
                    ))
                    fig_fi.update_layout(
                        height=max(250, len(feature_cols) * 30),
                        template='plotly_dark', paper_bgcolor='#1c1f23', plot_bgcolor='#09090b',
                        xaxis_title="Importance", margin=dict(l=0, r=0, t=10, b=0)
                    )
                    st.plotly_chart(fig_fi, width='stretch')

                with st.expander("Residual Analysis"):
                    residuals = y_te - preds
                    fig_res = px.histogram(residuals, nbins=60, color_discrete_sequence=['#2e5f65'],
                                          labels={'value': 'Residual (USD)'})
                    fig_res.update_layout(
                        height=250, template='plotly_dark', paper_bgcolor='#1c1f23', plot_bgcolor='#09090b',
                        xaxis_title="Residual (USD)", showlegend=False
                    )
                    st.plotly_chart(fig_res, width='stretch')


# ==============================================================================
# ABOUT
# ==============================================================================
elif page == "About":

    st.markdown("<h1>About This App</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#1c1f23; border:1px solid #2e5f65; border-radius:12px; padding:24px 28px; margin-bottom:20px;">
        <h3 style="color:#ffc72c; margin-top:0;">Gold Price Prediction App</h3>
        <p style="color:#e0f7fa; line-height:1.8;">
            This application provides an interactive dashboard and machine learning-based prediction
            tool for gold price analysis. It uses historical data from Yahoo Finance covering
            gold futures, crude oil, the US Dollar Index, and the S&P 500.
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
                <li>Key performance metrics</li>
                <li>RSI and Bollinger Bands indicators</li>
                <li>Asset comparison (normalized)</li>
                <li>Correlation heatmap</li>
                <li>Yearly average bar chart</li>
                <li>Volume analysis</li>
                <li>Daily return distribution</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background:#1c1f23; border:1px solid #2e5f65; border-radius:12px; padding:20px; margin-bottom:16px;">
            <h3 style="color:#ffc72c; margin-top:0;">ML Models</h3>
            <ul style="color:#e0f7fa; line-height:2; margin:0; padding-left:20px;">
                <li><b style="color:#ffc72c;">Random Forest</b> - ensemble of decision trees</li>
                <li><b style="color:#ffc72c;">Linear Regression</b> - baseline linear model</li>
                <li><b style="color:#ffc72c;">XGBoost</b> - gradient boosting</li>
                <li><b style="color:#ffc72c;">LSTM</b> - deep learning (local only)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#1c1f23; border:1px solid #2e5f65; border-radius:12px; padding:20px;">
            <h3 style="color:#ffc72c; margin-top:0;">Tech Stack</h3>
            <ul style="color:#e0f7fa; line-height:2; margin:0; padding-left:20px;">
                <li>Python 3.11</li>
                <li>Streamlit</li>
                <li>Plotly</li>
                <li>scikit-learn / XGBoost</li>
                <li>pandas / numpy</li>
                <li>GitHub Actions (CI/CD)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#1c1f23; border:1px solid #2e5f65; border-radius:12px; padding:20px; margin-top:4px;">
        <p style="color:#e0f7fa; margin:0; font-size:0.85rem; opacity:0.7; text-align:center;">
            Built with Streamlit - Data from Yahoo Finance - Auto-updated daily
        </p>
    </div>
    """, unsafe_allow_html=True)
