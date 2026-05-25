# 🥇 Gold Price Prediction

An interactive Streamlit web app for analyzing and predicting gold prices using machine learning.

**Built by:** Noor Alshorman

---

## 📊 Features

- **Dashboard** — Interactive charts for gold price history, comparison with oil, dollar index & S&P 500, correlation heatmap, yearly averages, and volume analysis
- **Prediction** — Train ML models (Random Forest, Linear Regression, XGBoost) with customizable lag features and evaluate performance with R², RMSE, MAE metrics

## 📁 Dataset

`TSDATA.csv` — Daily historical data from [Investing.com](https://investing.com) (1986–2025) including:

| Column | Description |
|---|---|
| Price_Gold | Gold closing price (USD/oz) |
| Price_Oil | Crude oil price |
| Price_Dollar | US Dollar Index |
| Price_Stocks | S&P 500 index |

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/2005noor27/gold-price-prediction.git
cd gold-price-prediction

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

## 🛠️ Tech Stack

- **Streamlit** — Web app framework
- **Plotly** — Interactive charts
- **scikit-learn** — ML models
- **XGBoost** — Gradient boosting
- **Pandas / NumPy** — Data processing
