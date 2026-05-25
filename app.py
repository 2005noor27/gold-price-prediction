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
.stApp { background-color: #0a0a0f; }
[data-testid="stSidebar"] { background-color: #0f0f1a; border-right: 1px solid #2a2a3d; }

/* Headings */
h1 { color: #FFD700 !important; letter-spacing: 1px; }
h2, h3 { color: #e0b830 !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #13131f, #1c1c2e);
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricValue"] { font-size: 1.5rem; color: #FFD700 !important; font-weight: 700; }
[data-testid="stMetricLabel"] { font-size: 0.82rem; color: #aaaacc; }
[data-testid="stMetricDelta"] { font-size: 0.85rem; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #b8860b, #FFD700);
    color: #0a0a0f;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    transition: all 0.2s ease;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }

/* Selectbox / Sliders */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stMultiSelect"] label { color: #c8c8e8 !important; font-size: 0.9rem; }

/* Divider */
hr { border-color: #2a2a4a; }

/* Expander */
[data-testid="stExpander"] { border: 1px solid #2a2a4a; border-radius: 8px; }

/* Sidebar text */
.stRadio label { color: #e0e0f0 !important; }
.block-container { padding-top: 1.5rem; }
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
    st.markdown("## 🥇 Gold Price App")
    st.markdown("---")
    page = st.radio("Navigate", ["📊 Dashboard", "🔮 Prediction"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### 📅 Date Range")
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()

    start_date = st.date_input("From", value=pd.to_datetime("2010-01-01").date(),
                               min_value=min_date, max_value=max_date)
    end_date   = st.date_input("To",   value=max_date,
                               min_value=min_date, max_value=max_date)

    st.markdown("---")
    st.caption(f"📦 Source: Investing.com")
    st.caption(f"🗓️ Period: 1986 – 2025")

# Filter
mask        = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
filtered_df = df[mask].copy()
st.sidebar.caption(f"📈 {len(filtered_df):,} trading days")


# ════════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":

    st.title("📊 Gold Price Dashboard")
    st.markdown(f"**{start_date}** → **{end_date}**  |  {len(filtered_df):,} trading days")

    # ── KPI Cards ───────────────────────────────────────────────────────────────
    gold_clean = filtered_df.dropna(subset=['Price_Gold'])
    if not gold_clean.empty:
        latest = gold_clean.iloc[-1]
        first  = gold_clean.iloc[0]
        delta  = latest['Price_Gold'] - first['Price_Gold']
        pct    = (delta / first['Price_Gold']) * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price",  f"${latest['Price_Gold']:,.2f}", f"{delta:+,.2f}  ({pct:+.1f}%)")
        c2.metric("Period High",    f"${gold_clean['High_Gold'].max():,.2f}")
        c3.metric("Period Low",     f"${gold_clean['Low_Gold'].min():,.2f}")
        c4.metric("Average Price",  f"${gold_clean['Price_Gold'].mean():,.2f}")

    st.markdown("---")

    # ── Gold Price Chart ─────────────────────────────────────────────────────────
    st.subheader("📈 Gold Price Over Time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered_df['Date'], y=filtered_df['Price_Gold'],
        mode='lines', name='Close',
        line=dict(color='#FFD700', width=2),
        fill='tozeroy', fillcolor='rgba(255,215,0,0.08)'
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
        height=440, template='plotly_dark',
        xaxis_title="Date", yaxis_title="Price (USD)",
        hovermode='x unified',
        legend=dict(orientation='h', y=1.02, x=0)
    )
    st.plotly_chart(fig, use_container_width=True)

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
            palette = {'Gold':'#FFD700','Oil':'#4CAF50',
                       'Dollar Index':'#2196F3','S&P 500':'#FF5722'}
            for col, color in palette.items():
                fig2.add_trace(go.Scatter(x=norm['Date'], y=norm[col],
                                          mode='lines', name=col,
                                          line=dict(color=color, width=1.5)))
            fig2.update_layout(height=340, template='plotly_dark',
                               yaxis_title="Normalized (0–100)",
                               hovermode='x unified',
                               legend=dict(orientation='h', y=1.02))
            st.plotly_chart(fig2, use_container_width=True)
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
            fig3.update_layout(height=340, template='plotly_dark')
            st.plotly_chart(fig3, use_container_width=True)
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
        fig4.update_layout(height=300, template='plotly_dark',
                           showlegend=False, xaxis_title="")
        st.plotly_chart(fig4, use_container_width=True)

    with r2:
        st.subheader("📦 Trading Volume")
        vol = filtered_df[['Date','Volume_Gold']].dropna()
        if not vol.empty:
            fig5 = go.Figure(go.Bar(
                x=vol['Date'], y=vol['Volume_Gold'],
                marker_color='rgba(255,215,0,0.55)'
            ))
            fig5.update_layout(height=300, template='plotly_dark',
                               yaxis_title="Volume", xaxis_title="")
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.info("No volume data in selected range.")

    # ── Daily Change Distribution ────────────────────────────────────────────────
    st.subheader("📉 Daily Change % Distribution")
    chg = filtered_df['Change%_Gold'].dropna() * 100
    if not chg.empty:
        fig6 = px.histogram(chg, nbins=80, color_discrete_sequence=['#FFD700'],
                            labels={'value': 'Daily Change (%)', 'count': 'Days'})
        fig6.update_layout(height=250, template='plotly_dark',
                           xaxis_title="Daily Change (%)", showlegend=False)
        st.plotly_chart(fig6, use_container_width=True)

    # ── Raw Data ────────────────────────────────────────────────────────────────
    with st.expander("📋 View Raw Data"):
        st.dataframe(
            filtered_df[['Date','Price_Gold','High_Gold','Low_Gold',
                         'Price_Oil','Price_Dollar','Price_Stocks']]
            .dropna().set_index('Date'),
            use_container_width=True
        )


# ════════════════════════════════════════════════════════════════════════════════
#  PREDICTION
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prediction":

    st.title("🔮 Gold Price Prediction")
    st.markdown("Train a machine learning model to predict gold prices from historical patterns.")

    # ── Config ──────────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        model_name = st.selectbox("🤖 Model", ["Random Forest", "Linear Regression", "XGBoost"])
    with c2:
        test_pct = st.slider("🧪 Test Size %", 10, 40, 20)
    with c3:
        n_lags = st.slider("⏳ Lag Features (days)", 1, 30, 5)

    extra_feats = st.multiselect(
        "📌 Additional Features",
        ['Price_Oil', 'Price_Dollar', 'Price_Stocks'],
        default=['Price_Oil', 'Price_Dollar']
    )

    run = st.button("🚀 Train Model", type="primary", use_container_width=True)

    if run:
        with st.spinner("Training … please wait"):

            # ── Prepare features ────────────────────────────────────────────────
            cols = ['Date', 'Price_Gold'] + extra_feats
            ml = filtered_df[cols].dropna().sort_values('Date').reset_index(drop=True)

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
                                     line=dict(color='rgba(255,215,0,0.3)', width=1)))
            fig.add_trace(go.Scatter(x=d_te, y=y_te, mode='lines', name='Actual (Test)',
                                     line=dict(color='#FFD700', width=2)))
            fig.add_trace(go.Scatter(x=d_te, y=y_pred, mode='lines', name='Predicted',
                                     line=dict(color='#FF4444', width=2, dash='dash')))
            fig.add_vline(x=d_te.iloc[0], line_dash='dash', line_color='gray',
                          annotation_text='Train | Test', annotation_position='top left')
            fig.update_layout(height=480, template='plotly_dark',
                              xaxis_title="Date", yaxis_title="Gold Price (USD)",
                              hovermode='x unified',
                              legend=dict(orientation='h', y=1.02))
            st.plotly_chart(fig, use_container_width=True)

            # ── Residuals ─────────────────────────────────────────────────────────
            st.subheader("📉 Residuals")
            resid = y_te.values - y_pred
            colors_r = ['#FF4444' if r < 0 else '#4CAF50' for r in resid]
            fig_r = go.Figure(go.Bar(x=d_te, y=resid, marker_color=colors_r))
            fig_r.add_hline(y=0, line_color='white', line_dash='dash')
            fig_r.update_layout(height=240, template='plotly_dark',
                                yaxis_title="Residual (USD)", xaxis_title="")
            st.plotly_chart(fig_r, use_container_width=True)

            # ── Feature Importance ────────────────────────────────────────────────
            if hasattr(model, 'feature_importances_'):
                st.subheader("🎯 Feature Importance")
                imp = pd.DataFrame({'Feature': feat_cols,
                                    'Importance': model.feature_importances_})\
                        .sort_values('Importance', ascending=True)
                fig_i = px.bar(imp, x='Importance', y='Feature', orientation='h',
                               color='Importance', color_continuous_scale='Oranges')
                fig_i.update_layout(height=max(300, len(feat_cols) * 25),
                                    template='plotly_dark', showlegend=False)
                st.plotly_chart(fig_i, use_container_width=True)

    else:
        st.info("👆 Choose your settings above and click **Train Model** to start.")
        st.subheader("📋 Latest Data")
        st.dataframe(
            filtered_df[['Date','Price_Gold','Price_Oil',
                         'Price_Dollar','Price_Stocks']]
            .dropna().tail(10).set_index('Date'),
            use_container_width=True
        )
