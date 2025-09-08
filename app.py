import streamlit as st
import requests
import pandas as pd
import altair as alt
import numpy as np
import itertools

# --- Fetch OHLCV data from CoinGecko ---
def get_coingecko_ohlcv(coin_id, vs_currency="usd"):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": "max", "interval": "daily"}
    response = requests.get(url, params=params)

    if response.status_code != 200:
        st.error(f"Error fetching {coin_id} from CoinGecko: {response.status_code}")
        return pd.DataFrame()

    data = response.json()
    if "prices" not in data:
        st.error(f"No price data found for {coin_id}.")
        return pd.DataFrame()

    prices = data["prices"]
    volumes = data["total_volumes"]
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["volume"] = [v[1] for v in volumes]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

# --- Momentum + probability trading signals ---
def add_signals(df, window=5, buy_thr=0.6, sell_thr=0.4):
    df = df.copy()
    df["return"] = df["price"].pct_change()
    df["momentum"] = df["price"].pct_change(periods=window)
    df["prob_up"] = 1 / (1 + np.exp(-10 * df["momentum"].fillna(0)))

    df["signal"] = 0
    df.loc[df["prob_up"] > buy_thr, "signal"] = 1   # Buy
    df.loc[df["prob_up"] < sell_thr, "signal"] = -1  # Sell
    return df

# --- Backtest strategy ---
def backtest(df):
    df = df.copy()
    df["next_return"] = df["price"].pct_change().shift(-1)
    trades = df[df["signal"] != 0]
    num_trades = len(trades)
    wins = trades[trades["next_return"] > 0]
    win_rate = len(wins) / num_trades if num_trades > 0 else 0
    avg_return = trades["next_return"].mean() if num_trades > 0 else 0

    df["strategy_return"] = df["signal"].shift(1) * df["return"]
    df["cumulative"] = (1 + df["strategy_return"].fillna(0)).cumprod()
    max_dd = ((df["cumulative"].cummax() - df["cumulative"]) / df["cumulative"].cummax()).max()

    summary = {
        "Total trades": num_trades,
        "Win rate": win_rate,
        "Average next-day return": avg_return,
        "Max drawdown": max_dd
    }
    return summary, df

# --- Auto-tune parameters ---
def tune_parameters(df, window_range=range(3,21), buy_range=np.arange(0.55,0.76,0.05), sell_range=np.arange(0.25,0.46,0.05)):
    best_params = None
    best_return = -np.inf
    
    for window, buy_thr, sell_thr in itertools.product(window_range, buy_range, sell_range):
        temp_df = df.copy()
        temp_df["momentum"] = temp_df["price"].pct_change(periods=window)
        temp_df["prob_up"] = 1 / (1 + np.exp(-10 * temp_df["momentum"].fillna(0)))
        
        temp_df["signal"] = 0
        temp_df.loc[temp_df["prob_up"] > buy_thr, "signal"] = 1
        temp_df.loc[temp_df["prob_up"] < sell_thr, "signal"] = -1
        
        temp_df["return"] = temp_df["price"].pct_change()
        temp_df["strategy_return"] = temp_df["signal"].shift(1) * temp_df["return"]
        cum_return = (1 + temp_df["strategy_return"].fillna(0)).prod()
        
        if cum_return > best_return:
            best_return = cum_return
            best_params = {"window": window, "buy_thr": buy_thr, "sell_thr": sell_thr}
    
    return best_params, best_return

# --- Streamlit UI ---
st.title("AI16Z & ADA Trading Signals (Auto-Tuned Strategy)")

coin_options = {
    "Cardano (ADA/USDT)": "cardano",
    "AI16Z (AI16Z/USDT)": "ai16z"
}

selected_label = st.selectbox("Choose a coin:", list(coin_options.keys()))
coin_choice = coin_options[selected_label]  # maps label -> API id

# Load data
df = get_coingecko_ohlcv(coin_choice)

if df.empty:
    st.warning("No data available for this coin.")
else:
    # Auto-tune parameters
    st.subheader("Optimizing strategy parameters...")
    best_params, best_return = tune_parameters(df)
    st.write("**Best parameters found:**", best_params)
    st.write(f"Cumulative return with these parameters: {best_return:.2f}x")

    # Apply best parameters
    df = add_signals(df, window=best_params["window"], buy_thr=best_params["buy_thr"], sell_thr=best_params["sell_thr"])
    summary, df = backtest(df)

    st.subheader(f"Data for {selected_label}")
    st.write(df.tail())

    # --- Price chart with buy/sell markers ---
    base = alt.Chart(df).encode(x="timestamp:T")
    price_line = base.mark_line().encode(y="price:Q")
    buy_markers = base.mark_point(color="green", size=80).encode(y="price:Q").transform_filter("datum.signal == 1")
    sell_markers = base.mark_point(color="red", size=80).encode(y="price:Q").transform_filter("datum.signal == -1")
    price_chart = (price_line + buy_markers + sell_markers).properties(title="Price with Buy (green) / Sell (red) Signals")
    st.altair_chart(price_chart, use_container_width=True)

    # --- Probability chart ---
    prob_chart = alt.Chart(df).mark_line(color="blue").encode(x="timestamp:T", y="prob_up:Q").properties(title="Probability of Next-Day Positive Return")
    st.altair_chart(prob_chart, use_container_width=True)

    # --- Cumulative returns chart ---
    cum_chart = alt.Chart(df).mark_line(color="purple").encode(x="timestamp:T", y="cumulative:Q").properties(title="Cumulative Strategy Returns")
    st.altair_chart(cum_chart, use_container_width=True)

    # --- Backtest summary ---
    st.subheader("Backtest Summary")
    for k, v in summary.items():
        if isinstance(v, float):
            st.write(f"**{k}:** {v:.2%}" if 'rate' in k.lower() or 'return' in k.lower() or 'drawdown' in k.lower() else f"**{k}:** {v}")
        else:
            st.write(f"**{k}:** {v}")
