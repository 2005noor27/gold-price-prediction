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

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gold Price Prediction",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Background */
.stApp { background-color: #00296b; }
[data-testid="stSidebar"] { background-color: #001d4e; border-right: 2px solid #00509d; }

/* Headings */
h1 { color: #ffd500 !important; letter-spacing: 1px; }
h2, h3 { color: #fdc500 !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #003f88, #00509d);
    border: 1px solid #00509d;
    border-top: 3px solid #fdc500;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricValue"] { font-size: 1.5rem; color: #ffd500 !important; font-weight: 700; }
[data-testid="stMetricLabel"] { font-size: 0.82rem; color: #a0c4ff; }
[data-testid="stMetricDelta"] { font-size: 0.85rem; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #fdc500, #ffd500);
    color: #00296b;
    font-weight: 800;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    transition: all 0.2s ease;
    letter-spacing: 0.5px;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); box-shadow: 0 4px 15px rgba(253,197,0,0.4); }

/* Selectbox / Sliders */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stMultiSelect"] label { color: #a0c4ff !important; font-size: 0.9rem; }

/* Divider */
hr { border-color: #00509d; }

/* Expander */
[data-testid="stExpander"] { border: 1px solid #00509d; border-radius: 8px; background: #003f88; }

/* Sidebar text */
.stRadio label { color: #e0f0ff !important; }
.block-container { padding-top: 1.5rem; }

/* Info/warning boxes */
[data-testid="stAlert"] { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading & Cleaning ────────────────────────────────────────────────────
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

    df['Year']  = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    return df


df = load_data()

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo / Header
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 8px 0;">
        <div style="font-size:3rem; line-height:1;">🥇</div>
        <div style="font-size:1.25rem; font-weight:800; color:#ffd500; letter-spacing:1px; margin-top:6px;">
            Gold Price
        </div>
        <div style="font-size:0.78rem; color:#a0c4ff; margin-top:2px; letter-spacing:2px;">
            PREDICTION APP
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#00509d; margin:8px 0 12px 0;'>", unsafe_allow_html=True)

    # Navigation
    page = st.radio(
        "nav",
        ["📊  Dashboard", "🔮  Prediction", "ℹ️  About"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#00509d; margin:12px 0;'>", unsafe_allow_html=True)

    # Date range (only show for Dashboard & Prediction)
    if page != "ℹ️  About":
        st.markdown("**📅 Date Range**")
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()

        start_date = st.date_input("From", value=pd.to_datetime("2010-01-01").date(),
                                   min_value=min_date, max_value=max_date)
        end_date   = st.date_input("To",   value=max_date,
                                   min_value=min_date, max_value=max_date)

        st.markdown("<hr style='border-color:#00509d; margin:12px 0;'>", unsafe_allow_html=True)
    else:
        start_date = pd.to_datetime("2010-01-01").date()
        end_date   = df['Date'].max().date()

    # Stats
    st.markdown(f"""
    <div style="font-size:0.78rem; color:#6090cc; line-height:2;">
        📦 &nbsp;Source: Investing.com<br>
        🗓️ &nbsp;Period: 1986 – 2025<br>
        📈 &nbsp;9,933 trading days
    </div>
    """, unsafe_allow_html=True)

# Filter
mask        = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
filtered_df = df[mask].copy()


# ════════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════
if page == "📊  Dashboard":

    st.markdown("""
    <h1 style="display:flex; align-items:center; gap:12px;">
        <span style="font-size:2.2rem;">📊</span>
        <span>Gold Price Dashboard</span>
    </h1>
    """, unsafe_allow_html=True)
    st.markdown(f"🗓️ **{start_date}** → **{end_date}**  &nbsp;|&nbsp;  📈 {len(filtered_df):,} trading days")

    # ── KPI Cards ───────────────────────────────────────────────────────────────
    gold_clean = filtered_df.dropna(subset=['Price_Gold'])
    if not gold_clean.empty:
        latest = gold_clean.iloc[-1]
        first  = gold_clean.iloc[0]
        delta  = latest['Price_Gold'] - first['Price_Gold']
        pct    = (delta / first['Price_Gold']) * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 Current Price",  f"${latest['Price_Gold']:,.2f}", f"{delta:+,.2f}  ({pct:+.1f}%)")
        c2.metric("🚀 Period High",    f"${gold_clean['High_Gold'].max():,.2f}")
        c3.metric("📉 Period Low",     f"${gold_clean['Low_Gold'].min():,.2f}")
        c4.metric("📊 Average Price",  f"${gold_clean['Price_Gold'].mean():,.2f}")

    st.markdown("---")

    # ── Gold Price Chart ─────────────────────────────────────────────────────────
    st.subheader("📈 Gold Price Over Time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered_df['Date'], y=filtered_df['Price_Gold'],
        mode='lines', name='Close',
        line=dict(color='#ffd500', width=2),
        fill='tozeroy', fillcolor='rgba(253,197,0,0.12)'
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
        height=440, template='plotly_dark', paper_bgcolor='#003f88', plot_bgcolor='#00296b',
        xaxis_title="Date", yaxis_title="Price (USD)",
        hovermode='x unified',
        legend=dict(orientation='h', y=1.02, x=0)
    )
    st.plotly_chart(fig, width='stretch')

    # ── Comparison & Correlation ─────────────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.subheader("📊 Asset Comparison (Normalized)")
        cmp = filtered_df[['Date','Price_Gold','Price_Oil',
                            'Price_Dollar','Price_Stocks']].dropna()
        if len(cmp) > 1:
            vals = MinMaxScaler((0, 100)).fit_transform(
                cmp[['Price_Gold','Price_Oil','Price_Dollar','Price_Stocks']]
            )
            norm = pd.DataFrame(vals, columns=['Gold','Oil','Dollar Index','S&P 500'])
            norm['Date'] = cmp['Date'].values

            fig2 = go.Figure()
            palette = {'Gold':'#ffd500','Oil':'#00d4aa',
                       'Dollar Index':'#60aaff','S&P 500':'#ff8c42'}
            for col, color in palette.items():
                fig2.add_trace(go.Scatter(x=norm['Date'], y=norm[col],
                                          mode='lines', name=col,
                                          line=dict(color=color, width=1.5)))
            fig2.update_layout(height=340, template='plotly_dark', paper_bgcolor='#003f88', plot_bgcolor='#00296b',
                               yaxis_title="Normalized (0–100)",
                               hovermode='x unified',
                               legend=dict(orientation='h', y=1.02))
            st.plotly_chart(fig2, width='stretch')
        else:
            st.info("Not enough data for comparison.")

    with right:
        st.subheader("🔥 Correlation Heatmap")
        corr_df = filtered_df[['Price_Gold','Price_Oil',
                                'Price_Dollar','Price_Stocks']].dropna()
        if len(corr_df) > 1:
            fig3 = px.imshow(
                corr_df.corr(),
                text_auto='.2f',
                color_continuous_scale='RdYlGn',
                zmin=-1, zmax=1
            )
            fig3.update_layout(height=340, template='plotly_dark', paper_bgcolor='#003f88', plot_bgcolor='#00296b')
            st.plotly_chart(fig3, width='stretch')
        else:
            st.info("Not enough data for correlation.")

    # ── Yearly Average & Volume ──────────────────────────────────────────────────
    l2, r2 = st.columns(2)

    with l2:
        st.subheader("📅 Yearly Average Gold Price")
        yearly = filtered_df.groupby('Year')['Price_Gold'].mean().reset_index()
        fig4 = px.bar(yearly, x='Year', y='Price_Gold',
                      color='Price_Gold', color_continuous_scale='Oranges',
                      labels={'Price_Gold': 'Avg Price (USD)'})
        fig4.update_layout(height=300, template='plotly_dark', paper_bgcolor='#003f88', plot_bgcolor='#00296b',
                           showlegend=False, xaxis_title="")
        st.plotly_chart(fig4, width='stretch')

    with r2:
        st.subheader("📦 Trading Volume")
        vol = filtered_df[['Date','Volume_Gold']].dropna()
        if not vol.empty:
            fig5 = go.Figure(go.Bar(
                x=vol['Date'], y=vol['Volume_Gold'],
                marker_color='rgba(253,197,0,0.7)'
            ))
            fig5.update_layout(height=300, template='plotly_dark', paper_bgcolor='#003f88', plot_bgcolor='#00296b',
                               yaxis_title="Volume", xaxis_title="")
            st.plotly_chart(fig5, width='stretch')
        else:
            st.info("No volume data in selected range.")

    # ── Daily Change Distribution ────────────────────────────────────────────────
    st.subheader("📉 Daily Change % Distribution")
    chg = filtered_df['Change%_Gold'].dropna() * 100
    if not chg.empty:
        fig6 = px.histogram(chg, nbins=80, color_discrete_sequence=['#fdc500'],
                            labels={'value': 'Daily Change (%)', 'count': 'Days'})
        fig6.update_layout(height=250, template='plotly_dark', paper_bgcolor='#003f88', plot_bgcolor='#00296b',
                           xaxis_title="Daily Change (%)", showlegend=False)
        st.plotly_chart(fig6, width='stretch')

    # ── Raw Data ────────────────────────────────────────────────────────────────
    with st.expander("📋 View Raw Data"):
        st.dataframe(
            filtered_df[['Date','Price_Gold','High_Gold','Low_Gold',
                         'Price_Oil','Price_Dollar','Price_Stocks']]
            .dropna().set_index('Date'),
            width='stretch'
        )


# ════════════════════════════════════════════════════════════════════════════════
#  PREDICTION
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Prediction":

    st.markdown("""
    <h1 style="display:flex; align-items:center; gap:12px;">
        <span style="font-size:2.2rem;">🔮</span>
        <span>Gold Price Prediction</span>
    </h1>
    """, unsafe_allow_html=True)
    st.markdown("Train a machine learning model to predict gold prices from historical patterns.")

    # ── Config ──────────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        model_name = st.selectbox("🤖 Model", ["Random Forest", "Linear Regression", "XGBoost", "LSTM (Deep Learning)"])
    with c2:
        test_pct = st.slider("🧪 Test Size %", 10, 40, 20)
    with c3:
        n_lags = st.slider("⏳ Lag Features (days)", 1, 30, 5)

    extra_feats = st.multiselect(
        "📌 Additional Features",
        ['Price_Oil', 'Price_Dollar', 'Price_Stocks'],
        default=['Price_Oil', 'Price_Dollar']
    )

    run = st.button("🚀 Train Model", type="primary", width='stretch')

    if run:
        with st.spinner("Training … please wait"):

            # ── Prepare features ────────────────────────────────────────────────
            cols = ['Date', 'Price_Gold'] + extra_feats
            ml = filtered_df[cols].dropna().sort_values('Date').reset_index(drop=True)

            # ── LSTM path (separate flow) ────────────────────────────────────────
            if model_name == "LSTM (Deep Learning)":
                try:
                    import tensorflow as tf
                    from tensorflow.keras.models import Sequential
                    from tensorflow.keras.layers import LSTM, Dense, Dropout
                    from tensorflow.keras.callbacks import EarlyStopping

                    tf.random.set_seed(42)

                    # Scale
                    feature_cols_lstm = ['Price_Gold'] + extra_feats
                    scaler_lstm = MinMaxScaler()
                    scaled = scaler_lstm.fit_transform(ml[feature_cols_lstm])

                    # Build sequences
                    SEQ_LEN = n_lags
                    X_seq, y_seq = [], []
                    for i in range(SEQ_LEN, len(scaled)):
                        X_seq.append(scaled[i - SEQ_LEN:i])
                        y_seq.append(scaled[i, 0])   # Price_Gold is index 0

                    X_seq = np.array(X_seq)
                    y_seq = np.array(y_seq)
                    dates_seq = ml['Date'].iloc[SEQ_LEN:].reset_index(drop=True)

                    split = int(len(X_seq) * (1 - test_pct / 100))
                    X_tr_l, X_te_l = X_seq[:split], X_seq[split:]
                    y_tr_l, y_te_l = y_seq[:split], y_seq[split:]
                    d_te           = dates_seq.iloc[split:]
                    y_te_orig      = ml['Price_Gold'].iloc[SEQ_LEN + split:].values

                    # Build LSTM model
                    lstm_model = Sequential([
                        LSTM(64, return_sequences=True,
                             input_shape=(SEQ_LEN, X_seq.shape[2])),
                        Dropout(0.2),
                        LSTM(32, return_sequences=False),
                        Dropout(0.2),
                        Dense(16, activation='relu'),
                        Dense(1)
                    ])
                    lstm_model.compile(optimizer='adam', loss='mse')

                    es = EarlyStopping(monitor='val_loss', patience=5,
                                       restore_best_weights=True)
                    with st.spinner("🧠 Training LSTM — this may take a minute…"):
                        lstm_model.fit(
                            X_tr_l, y_tr_l,
                            epochs=50, batch_size=32,
                            validation_split=0.1,
                            callbacks=[es], verbose=0
                        )

                    # Predict & inverse-scale
                    pred_scaled = lstm_model.predict(X_te_l, verbose=0).flatten()
                    dummy = np.zeros((len(pred_scaled), len(feature_cols_lstm)))
                    dummy[:, 0] = pred_scaled
                    y_pred = scaler_lstm.inverse_transform(dummy)[:, 0]
                    y_te   = pd.Series(y_te_orig)

                    r2   = r2_score(y_te, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
                    mae  = mean_absolute_error(y_te, y_pred)
                    mape = np.mean(np.abs((y_te.values - y_pred) / y_te.values)) * 100

                    st.markdown("### 📊 Model Performance")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("R² Score", f"{r2:.4f}")
                    m2.metric("RMSE",     f"${rmse:,.2f}")
                    m3.metric("MAE",      f"${mae:,.2f}")
                    m4.metric("MAPE",     f"{mape:.2f}%")
                    st.markdown("---")

                    # Chart
                    st.subheader("📈 Actual vs Predicted (LSTM)")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=ml['Date'], y=ml['Price_Gold'],
                        mode='lines', name='Full History',
                        line=dict(color='rgba(253,197,0,0.25)', width=1)
                    ))
                    fig.add_trace(go.Scatter(
                        x=d_te, y=y_te,
                        mode='lines', name='Actual (Test)',
                        line=dict(color='#ffd500', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=d_te, y=y_pred,
                        mode='lines', name='LSTM Predicted',
                        line=dict(color='#a78bfa', width=2, dash='dash')
                    ))
                    fig.add_vline(x=d_te.iloc[0], line_dash='dash',
                                  line_color='gray',
                                  annotation_text='Train | Test',
                                  annotation_position='top left')
                    fig.update_layout(
                        height=480, template='plotly_dark', paper_bgcolor='#003f88', plot_bgcolor='#00296b',
                        xaxis_title="Date", yaxis_title="Gold Price (USD)",
                        hovermode='x unified',
                        legend=dict(orientation='h', y=1.02)
                    )
                    st.plotly_chart(fig, width='stretch')

                    # Residuals
                    st.subheader("📉 Residuals")
                    resid = y_te.values - y_pred
                    colors_r = ['#FF4444' if r < 0 else '#4CAF50' for r in resid]
                    fig_r = go.Figure(go.Bar(x=d_te, y=resid,
                                            marker_color=colors_r))
                    fig_r.add_hline(y=0, line_color='white', line_dash='dash')
                    fig_r.update_layout(height=240, template='plotly_dark', paper_bgcolor='#003f88', plot_bgcolor='#00296b',
                                        yaxis_title="Residual (USD)", xaxis_title="")
                    st.plotly_chart(fig_r, width='stretch')

                    st.info("💡 LSTM uses sequential windows — feature importance is not available for deep learning models.")

                except ImportError:
                    st.warning("⚠️ LSTM requires TensorFlow which isn't available on this server (Python 3.14 not yet supported). Run locally with `pip install tensorflow` to use this model.")
                    st.info("💡 Try **Random Forest** or **XGBoost** instead — they give excellent results on this dataset!")

                st.stop()   # skip the ML flow below for LSTM

            # ── ML path (non-LSTM) ───────────────────────────────────────────────
            for lag in range(1, n_lags + 1):
                ml[f'Lag_{lag}'] = ml['Price_Gold'].shift(lag)

            ml['MA_7']  = ml['Price_Gold'].rolling(7).mean()
            ml['MA_30'] = ml['Price_Gold'].rolling(30).mean()
            ml['MA_90'] = ml['Price_Gold'].rolling(90).mean()
            ml = ml.dropna().reset_index(drop=True)

            feat_cols = [f'Lag_{i}' for i in range(1, n_lags + 1)] \
                        + ['MA_7', 'MA_30', 'MA_90'] + extra_feats

            X, y, dates = ml[feat_cols], ml['Price_Gold'], ml['Date']
            split = int(len(X) * (1 - test_pct / 100))
            X_tr, X_te = X.iloc[:split], X.iloc[split:]
            y_tr, y_te = y.iloc[:split], y.iloc[split:]
            d_te        = dates.iloc[split:]

            # ── Train ────────────────────────────────────────────────────────────
            if model_name == "Linear Regression":
                sc = StandardScaler()
                X_tr = sc.fit_transform(X_tr)
                X_te = sc.transform(X_te)
                model = LinearRegression()

            elif model_name == "XGBoost":
                try:
                    from xgboost import XGBRegressor
                    model = XGBRegressor(n_estimators=200, learning_rate=0.05,
                                        max_depth=6, random_state=42,
                                        verbosity=0)
                except ImportError:
                    st.warning("XGBoost not found – switching to Random Forest.")
                    model = RandomForestRegressor(n_estimators=100, random_state=42)

            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)

            # ── Metrics ──────────────────────────────────────────────────────────
            r2   = r2_score(y_te, y_pred)
            rmse = np.sqrt(mean_squared_error(y_te, y_pred))
            mae  = mean_absolute_error(y_te, y_pred)
            mape = np.mean(np.abs((y_te.values - y_pred) / y_te.values)) * 100

            st.markdown("### 📊 Model Performance")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("R² Score", f"{r2:.4f}")
            m2.metric("RMSE",     f"${rmse:,.2f}")
            m3.metric("MAE",      f"${mae:,.2f}")
            m4.metric("MAPE",     f"{mape:.2f}%")

            st.markdown("---")

            # ── Actual vs Predicted ───────────────────────────────────────────────
            st.subheader("📈 Actual vs Predicted")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=y, mode='lines', name='Full History',
                                     line=dict(color='rgba(253,197,0,0.3)', width=1)))
            fig.add_trace(go.Scatter(x=d_te, y=y_te, mode='lines', name='Actual (Test)',
                                     line=dict(color='#ffd500', width=2)))
            fig.add_trace(go.Scatter(x=d_te, y=y_pred, mode='lines', name='Predicted',
                                     line=dict(color='#FF4444', width=2, dash='dash')))
            fig.add_vline(x=d_te.iloc[0], line_dash='dash', line_color='gray',
                          annotation_text='Train | Test', annotation_position='top left')
            fig.update_layout(height=480, template='plotly_dark', paper_bgcolor='#003f88', plot_bgcolor='#00296b',
                              xaxis_title="Date", yaxis_title="Gold Price (USD)",
                              hovermode='x unified',
                              legend=dict(orientation='h', y=1.02))
            st.plotly_chart(fig, width='stretch')

            # ── Residuals ─────────────────────────────────────────────────────────
            st.subheader("📉 Residuals")
            resid = y_te.values - y_pred
            colors_r = ['#FF4444' if r < 0 else '#4CAF50' for r in resid]
            fig_r = go.Figure(go.Bar(x=d_te, y=resid, marker_color=colors_r))
            fig_r.add_hline(y=0, line_color='white', line_dash='dash')
            fig_r.update_layout(height=240, template='plotly_dark', paper_bgcolor='#003f88', plot_bgcolor='#00296b',
                                yaxis_title="Residual (USD)", xaxis_title="")
            st.plotly_chart(fig_r, width='stretch')

            # ── Feature Importance ────────────────────────────────────────────────
            if hasattr(model, 'feature_importances_'):
                st.subheader("🎯 Feature Importance")
                imp = pd.DataFrame({'Feature': feat_cols,
                                    'Importance': model.feature_importances_})\
                        .sort_values('Importance', ascending=True)
                fig_i = px.bar(imp, x='Importance', y='Feature', orientation='h',
                               color='Importance', color_continuous_scale='Oranges')
                fig_i.update_layout(height=max(300, len(feat_cols) * 25),
                                    template='plotly_dark', paper_bgcolor='#003f88', plot_bgcolor='#00296b', showlegend=False)
                st.plotly_chart(fig_i, width='stretch')

    else:
        st.info("👆 Choose your settings above and click **Train Model** to start.")
        st.subheader("📋 Latest Data")
        st.dataframe(
            filtered_df[['Date','Price_Gold','Price_Oil',
                         'Price_Dollar','Price_Stocks']]
            .dropna().tail(10).set_index('Date'),
            width='stretch'
        )


# ════════════════════════════════════════════════════════════════════════════════
#  ABOUT
# ════════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️  About":

    st.markdown("""
    <h1 style="display:flex; align-items:center; gap:12px;">
        <span style="font-size:2.2rem;">🥇</span>
        <span>About This Project</span>
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #003f88, #00509d);
        border: 1px solid #00509d;
        border-left: 4px solid #fdc500;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
    ">
        <p style="color:#e0e0f0; font-size:1.05rem; line-height:1.8; margin:0;">
            This app analyzes <strong style="color:#ffd500;">gold price data</strong> spanning nearly
            four decades (1986–2025) and builds machine learning models to predict future prices.
            It covers the relationship between gold and key global indicators: crude oil,
            the US dollar index, and the S&P 500.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Cards row
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div style="background:#003f88; border:1px solid #00509d; border-radius:12px; padding:20px; text-align:center;">
            <div style="font-size:2.5rem;">📅</div>
            <div style="color:#ffd500; font-size:1.6rem; font-weight:800; margin:8px 0;">39 Years</div>
            <div style="color:#a0c4ff; font-size:0.85rem;">of daily market data<br>1986 – 2025</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div style="background:#003f88; border:1px solid #00509d; border-radius:12px; padding:20px; text-align:center;">
            <div style="font-size:2.5rem;">📈</div>
            <div style="color:#ffd500; font-size:1.6rem; font-weight:800; margin:8px 0;">9,933</div>
            <div style="color:#a0c4ff; font-size:0.85rem;">trading days<br>in the dataset</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div style="background:#003f88; border:1px solid #00509d; border-radius:12px; padding:20px; text-align:center;">
            <div style="font-size:2.5rem;">🤖</div>
            <div style="color:#ffd500; font-size:1.6rem; font-weight:800; margin:8px 0;">3 Models</div>
            <div style="color:#a0c4ff; font-size:0.85rem;">Random Forest · XGBoost<br>Linear Regression</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Dataset section
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### 📦 Dataset")
        st.markdown("""
        <div style="background:#003f88; border:1px solid #00509d; border-radius:10px; padding:16px;">
        <table style="width:100%; color:#c0c0e0; font-size:0.88rem; border-collapse:collapse;">
            <tr style="border-bottom:1px solid #00509d;">
                <td style="padding:8px; color:#ffd500; font-weight:600;">Column</td>
                <td style="padding:8px; color:#ffd500; font-weight:600;">Description</td>
            </tr>
            <tr style="border-bottom:1px solid #003f88;">
                <td style="padding:8px;">🥇 Price_Gold</td>
                <td style="padding:8px;">Gold closing price (USD/oz)</td>
            </tr>
            <tr style="border-bottom:1px solid #003f88;">
                <td style="padding:8px;">🛢️ Price_Oil</td>
                <td style="padding:8px;">Crude oil price (USD)</td>
            </tr>
            <tr style="border-bottom:1px solid #003f88;">
                <td style="padding:8px;">💵 Price_Dollar</td>
                <td style="padding:8px;">US Dollar Index (DXY)</td>
            </tr>
            <tr>
                <td style="padding:8px;">📈 Price_Stocks</td>
                <td style="padding:8px;">S&P 500 index</td>
            </tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("### 🛠️ Tech Stack")
        tech = [
            ("🌐", "Streamlit",    "Web app framework"),
            ("📊", "Plotly",       "Interactive charts"),
            ("🤖", "scikit-learn", "ML models"),
            ("⚡", "XGBoost",      "Gradient boosting"),
            ("🐼", "Pandas",       "Data processing"),
            ("🔢", "NumPy",        "Numerical computing"),
        ]
        for icon, name, desc in tech:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:12px;
                        background:#003f88; border:1px solid #00509d;
                        border-radius:8px; padding:10px 14px; margin-bottom:6px;">
                <span style="font-size:1.3rem;">{icon}</span>
                <div>
                    <span style="color:#ffd500; font-weight:600;">{name}</span>
                    <span style="color:#6090cc; font-size:0.82rem; margin-left:8px;">{desc}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; padding:20px; border-top:1px solid #00509d; margin-top:16px;">
        <span style="color:#6090cc; font-size:0.85rem;">
            Built by &nbsp;<strong style="color:#ffd500;">Noor Alshorman</strong>
            &nbsp;·&nbsp; Data from
            <a href="https://investing.com" target="_blank"
               style="color:#ffd500; text-decoration:none;">Investing.com</a>
            &nbsp;·&nbsp;
            <a href="https://github.com/2005noor27/gold-price-prediction" target="_blank"
               style="color:#ffd500; text-decoration:none;">GitHub ↗</a>
        </span>
    </div>
    """, unsafe_allow_html=True)
