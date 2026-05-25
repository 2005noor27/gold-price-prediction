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
            fig5 =