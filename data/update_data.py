# -*- coding: utf-8 -*-
"""
update_data.py
==============
يجيب بيانات الذهب والنفط والدولار والأسهم من آخر تاريخ بالملف لليوم
ثم يضيفها على TSDATA.csv تلقائياً.

الاستخدام:
    pip install yfinance pandas
    python update_data.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

# ── إعدادات ─────────────────────────────────────────────────────────────────────
CSV_PATH = Path(__file__).parent / "TSDATA.csv"

TICKERS = {
    "gold":   "GC=F",       # Gold Futures
    "oil":    "CL=F",       # Crude Oil Futures
    "dollar": "DX-Y.NYB",   # US Dollar Index
    "sp500":  "^GSPC",      # S&P 500
}

# ── تنظيف الأعداد ────────────────────────────────────────────────────────────────
def clean_currency(x):
    if isinstance(x, str):
        return float(x.replace(",", "").replace("$", "").strip())
    return x

def clean_volume(x):
    if isinstance(x, str):
        x = x.strip().upper()
        if x in ["", "-", "N/A"]:
            return np.nan
        if "K" in x:
            return float(x.replace("K", "")) * 1_000
        elif "M" in x:
            return float(x.replace("M", "")) * 1_000_000
        elif "B" in x:
            return float(x.replace("B", "")) * 1_000_000_000
        return float(x.replace(",", ""))
    return x

# ── قراءة الملف الحالي ───────────────────────────────────────────────────────────
print("📂 Reading TSDATA.csv ...")
df_old = pd.read_csv(CSV_PATH)
df_old["Date"] = pd.to_datetime(df_old["Date"], format="mixed")
df_old = df_old.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)

# أعيدي كتابة الملف بصيغة تاريخ موحدة
df_old["Date"] = df_old["Date"].apply(lambda d: f"{d.month}/{d.day}/{d.year}")
df_old.to_csv(CSV_PATH, index=False)
print(f"   Fixed & cleaned: {len(df_old)} rows")

df_old["Date"] = pd.to_datetime(df_old["Date"], format="mixed")
last_date = df_old["Date"].max()
start_date = last_date + timedelta(days=1)
today = datetime.today()

print(f"   Last date in file : {last_date.date()}")
print(f"   Downloading from  : {start_date.date()} → {today.date()}")

if start_date.date() >= today.date():
    print("✅ Already up to date! Nothing to add.")
    exit()

# ── جلب البيانات ─────────────────────────────────────────────────────────────────
def download(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=False, progress=False)
        df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        print(f"   ⚠️  {ticker}: {e}")
        return pd.DataFrame()

print("\n📡 Fetching data ...")
gold   = download(TICKERS["gold"],   start_date, today + timedelta(days=1))
oil    = download(TICKERS["oil"],    start_date, today + timedelta(days=1))
dollar = download(TICKERS["dollar"], start_date, today + timedelta(days=1))
sp500  = download(TICKERS["sp500"],  start_date, today + timedelta(days=1))

if gold.empty:
    print("❌ Could not fetch Gold data. Check internet connection.")
    exit()

print(f"   Gold   rows: {len(gold)}")
print(f"   Oil    rows: {len(oil)}")
print(f"   Dollar rows: {len(dollar)}")
print(f"   S&P    rows: {len(sp500)}")

# ── بناء الـ DataFrame الجديد ────────────────────────────────────────────────────
print("\n🔨 Building new rows ...")

new_rows = pd.DataFrame(index=gold.index)
new_rows.index = pd.to_datetime(new_rows.index)

# Gold
new_rows["Price_Gold"]   = gold["Close"].values if "Close" in gold.columns else np.nan
new_rows["High_Gold"]    = gold["High"].values  if "High"  in gold.columns else np.nan
new_rows["Low_Gold"]     = gold["Low"].values   if "Low"   in gold.columns else np.nan
new_rows["Open_Gold"]    = gold["Open"].values  if "Open"  in gold.columns else np.nan
new_rows["Volume_Gold"]  = gold["Volume"].values if "Volume" in gold.columns else np.nan

# Change% Gold
new_rows["Change%_Gold"] = new_rows["Price_Gold"].pct_change() * 100
# Format as string with %
new_rows["Change%_Gold"] = new_rows["Change%_Gold"].apply(
    lambda x: f"{x:.2f}%" if pd.notna(x) else ""
)

# Oil
if not oil.empty and "Close" in oil.columns:
    oil_aligned = oil["Close"].reindex(new_rows.index)
    new_rows["Price_Oil"] = oil_aligned.values
else:
    new_rows["Price_Oil"] = np.nan

# Dollar
if not dollar.empty:
    d_aligned = dollar.reindex(new_rows.index)
    new_rows["Price_Dollar"]  = d_aligned["Close"].values  if "Close"  in dollar.columns else np.nan
    new_rows["High_Dollar"]   = d_aligned["High"].values   if "High"   in dollar.columns else np.nan
    new_rows["Low_Dollar"]    = d_aligned["Low"].values    if "Low"    in dollar.columns else np.nan
    new_rows["Open_Dollar"]   = d_aligned["Open"].values   if "Open"   in dollar.columns else np.nan
    new_rows["Volume_Dollar"] = 0
else:
    for c in ["Price_Dollar","High_Dollar","Low_Dollar","Open_Dollar","Volume_Dollar"]:
        new_rows[c] = np.nan

# S&P 500
if not sp500.empty:
    s_aligned = sp500.reindex(new_rows.index)
    new_rows["Price_Stocks"]  = s_aligned["Close"].values  if "Close"  in sp500.columns else np.nan
    new_rows["High_Stocks"]   = s_aligned["High"].values   if "High"   in sp500.columns else np.nan
    new_rows["Low_Stocks"]    = s_aligned["Low"].values    if "Low"    in sp500.columns else np.nan
    new_rows["Open_Stocks"]   = s_aligned["Open"].values   if "Open"   in sp500.columns else np.nan
    new_rows["Volume_Stocks"] = s_aligned["Volume"].values if "Volume" in sp500.columns else np.nan
else:
    for c in ["Price_Stocks","High_Stocks","Low_Stocks","Open_Stocks","Volume_Stocks"]:
        new_rows[c] = np.nan

# رتبي الأعمدة بنفس ترتيب الملف الأصلي
new_rows = new_rows.reset_index().rename(columns={"index": "Date", "Datetime": "Date"})
# Cross-platform date format (M/D/YYYY without zero padding)
new_rows["Date"] = pd.to_datetime(new_rows["Date"]).apply(
    lambda d: f"{d.month}/{d.day}/{d.year}"
)

cols = [
    "Date",
    "Price_Gold","High_Gold","Low_Gold","Open_Gold","Volume_Gold","Change%_Gold",
    "Price_Oil",
    "Price_Dollar","High_Dollar","Low_Dollar","Open_Dollar","Volume_Dollar",
    "Price_Stocks","High_Stocks","Low_Stocks","Open_Stocks","Volume_Stocks"
]
new_rows = new_rows[[c for c in cols if c in new_rows.columns]]

# ── اضيفي الصفوف الجديدة ─────────────────────────────────────────────────────────
print(f"\n✅ Appending {len(new_rows)} new rows to TSDATA.csv ...")

new_rows.to_csv(CSV_PATH, mode="a", header=False, index=False)

# تحقق
df_check = pd.read_csv(CSV_PATH)
print(f"   Old rows : {len(df_old)}")
print(f"   New total: {len(df_check)}")
print(f"   Last date: {df_check['Date'].iloc[-1]}")
print("\n🎉 Done! TSDATA.csv is now up to date.")
