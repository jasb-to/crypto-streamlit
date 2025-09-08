import streamlit as st
import requests
import pandas as pd
import altair as alt
import numpy as np
import time

st.title("AI16Z (Solana) & ADA Trading Signals")

# --- Configuration ---
AI16Z_POOL_ID = "7qAVrzrbULwg1B13YseqA95Uapf8EVp9jQE5uipqFMoP"
GECKO_URL = f"https://api.geckoterminal.com/api/v2/networks/solana/pools/{AI16Z_POOL_ID}/market_chart"
ADA_COINGECKO_ID = "cardano"
VS_CURRENCY = "usd"

# --- Fetch AI16Z OHLCV from GeckoTerminal ---
def fetch_ai16z_geckoterminal():
    try:
        params = {"interval": "1d", "days": 365}  # daily for last year
        response = requests.get(GECKO_URL, params=params)
        if response.status_code != 200:
            st.warning(f"GeckoTerminal API error: {response.status_code}")
            return pd.DataFrame()
        data = response.json()
        if "data" not in data or "chart" not in data["data"]:
            st.warning("No chart data in GeckoTerminal response")
            return pd.DataFrame()
        chart = data["data"]["chart"]
        df = pd.DataFrame(chart)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df["price"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        return df[["timestamp","price","volume"]]
    except Exception as e:
        st.error(f"Error fetching AI16Z data: {e}")
        return pd.DataFrame()

# --- Fetch ADA from CoinGecko as fallback ---
def fetch_ada_coingecko():
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{ADA_COINGECKO_ID}/market_chart"
        params = {"vs_currency": VS_CURRENCY, "days": "max", "interval": "daily"}
        response = requests.get(url, params=params)
        if response.status_code != 200:
            st.warning(f"CoinGecko API error: {response.status_code}")
            return pd.DataFrame()
        data = response.json()
        df = pd.DataFrame(data["prices"], columns=["timestamp","price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["volume"] = [v[1] for v in data.get("total_volumes", [[0,0]]*len(df))]
        return df[["timestamp","price","volume"]]
    except Exception as e:
        st.error(f"Error fetching ADA data: {e}")
        return pd.DataFrame()

# --- Signal calculation ---
def add_signals(df, window=5, buy_thr=0.6, sell_thr=0.4):
    df = df.copy()
    df["return"] = df["price"].pct_change()
    df["momentum"] = df["price"].pct_change(periods=window)
    df["prob_up"] = 1 / (1 + np.exp(-10 * df["momentum"].fillna(0)))
    df["signal"] = 0
    df.loc[df["prob_up"] > buy_thr, "signal"] = 1
    df.loc[df["prob_up"] < sell_thr, "signal"] = -1
    return df

# --- Backtest ---
def backtest(df):
    df = df.copy()
    df["strategy_return"] = df["signal"].shift(1) * df["return"]
    df["cumulative"] = (1 + df["strategy_return"].fillna(0)).cumprod()
    return df

# --- Main execution ---
df = fetch_ai16z_geckoterminal()
if df.empty:
    st.info("Falling back to ADA data from CoinGecko...")
    df = fetch_ada_coingecko()
    if df.empty:
        st.warning("No data available for AI16Z or ADA.")
        st.stop()
    else:
        coin_label = "ADA"
else:
    coin_label = "AI16Z"

df = add_signals(df)
df = backtest(df)

st.subheader(f"{coin_label} Price & Signals")
base = alt.Chart(df).encode(x="timestamp:T")
price_line = base.mark_line().encode(y="price:Q")
buy_markers = base.mark_point(color="green", size=80).encode(y="price:Q").transform_filter("datum.signal == 1")
sell_markers = base.mark_point(color="red", size=80).encode(y="price:Q").transform_filter("datum.signal == -1")
st.altair_chart(price_line + buy_markers + sell_markers, use_container_width=True)

st.subheader("Cumulative Strategy Returns")
cum_chart = alt.Chart(df).mark_line(color="purple").encode(
    x="timestamp:T",
    y="cumulative:Q"
).properties(title=f"{coin_label} Strategy Cumulative Returns")
st.altair_chart(cum_chart, use_container_width=True)
