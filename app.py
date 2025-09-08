import streamlit as st
import requests
import pandas as pd
import altair as alt
import numpy as np

st.title("ADA Trading Signals (CoinGecko Free API)")

# --- Config ---
COIN_ID = "cardano"  # Cardano / ADA
VS_CURRENCY = "usd"

# --- Fetch OHLCV from CoinGecko ---
def fetch_coingecko_data(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": VS_CURRENCY, "days": "max", "interval": "daily"}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        st.error(f"Error fetching {coin_id}: {response.status_code}")
        return pd.DataFrame()
    data = response.json()
    if "prices" not in data:
        st.error(f"No price data for {coin_id}")
        return pd.DataFrame()
    df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    df["volume"] = [v[1] for v in data.get("total_volumes", [[0,0]]*len(df))]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

# --- Signals ---
def add_signals(df, window=5, buy_thr=0.6, sell_thr=0.4):
    df = df.copy()
    df["return"] = df["price"].pct_change()
    df["momentum"] = df["price"].pct_change(periods=window)
    df["prob_up"] = 1 / (1 + np.exp(-10 * df["momentum"].fillna(0)))
    df["signal"] = 0
    df.loc[df["prob_up"] > buy_thr, "signal"] = 1
    df.loc[df["prob_up"] < sell_thr, "signal"] = -1
    return df

def backtest(df):
    df = df.copy()
    df["strategy_return"] = df["signal"].shift(1) * df["return"]
    df["cumulative"] = (1 + df["strategy_return"].fillna(0)).cumprod()
    return df

# --- Run ---
df = fetch_coingecko_data(COIN_ID)

if df.empty:
    st.warning("No data available.")
else:
    df = add_signals(df)
    df = backtest(df)

    # Price + signals
    base = alt.Chart(df).encode(x="timestamp:T")
    price_line = base.mark_line().encode(y="price:Q")
    buy_markers = base.mark_point(color="green", size=80).encode(y="price:Q").transform_filter("datum.signal == 1")
    sell_markers = base.mark_point(color="red", size=80).encode(y="price:Q").transform_filter("datum.signal == -1")
    st.altair_chart(price_line + buy_markers + sell_markers, use_container_width=True)

    # Cumulative returns
    cum_chart = alt.Chart(df).mark_line(color="purple").encode(
        x="timestamp:T",
        y="cumulative:Q"
    ).properties(title="Cumulative Strategy Returns")
    st.altair_chart(cum_chart, use_container_width=True)
