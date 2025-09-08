import streamlit as st
import requests
import pandas as pd
import altair as alt
import numpy as np
import itertools

# --- Fetch OHLCV data from Binance ---
def get_binance_ohlcv(symbol="ADAUSDT", interval="1d"):
    url = "https://api4.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval}
    response = requests.get(url, params=params)

    if response.status_code != 200:
        st.error(f"Error fetching {symbol} from Binance: {response.status_code}")
        return pd.DataFrame()

    data = response.json()
    if not data:
        st.error(f"No OHLCV data for {symbol}.")
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_asset_volume","num_trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["price"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df[["open_time", "price", "volume"]]

# --- Momentum + probability trading signals ---
def add_signals(df, window=5, buy_thr=0.6, sell_thr=0.4):
    df = df.copy()
    df["return"] = df["price"].pct_change()
    df["momentum"] = df["price"].pct_change(periods=window)
    df["prob_up"] = 1 / (1 + np.exp(-10 * df["momentum"].fillna(0)))

    df["signal"] = 0
    df.loc[df["prob_up"] > buy_thr, "signal"] = 1
    df.loc[df["prob_up"] < sell_thr, "signal"] = -1
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
st.title("ADA/USDT Trading Signals (Binance Data)")

coin_options = {
    "Cardano (ADA/USDT)": "ADAUSDT",
    # AI16Z is not on Binance, so it will fail if selected
    "AI16Z (Not on Binance)": None
}

selected_label = st.selectbox("Choose a coin:", list(coin_options.keys()))
symbol = coin_options[selected_label]

if symbol is None:
    st.error("AI16Z is not listed on Binance, so no OHLCV data is available.")
else:
    # Load data
    df = get_binance_ohlcv(symbol)

    if df.empty:
        st.warning("No data available for this symbol.")
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
        base = alt.Chart(df).encode(x="open_time:T")
        price_line = base.mark_line().encode(y="price:Q")
        buy_markers = base.mark_point(color="green", size=80).encode(y="price:Q").transform_filter("datum.signal == 1")
        sell_markers = base.mark_point(color="red", size=80).encode(y="price:Q").transform_filter("datum.signal == -1")
        price_chart = (price_line + buy_markers + sell_markers).properties(title="Price with Buy (green) / Sell (red) Signals")
        st.altair_chart(price_chart, use_container_width=True)

        # --- Probability chart ---
        prob_chart = alt.Chart(df).mark_line(color="blue").encode(x="open_time:T", y="prob_up:Q").properties(title="Probability of Next-Day Positive Return")
        st.altair_chart(prob_chart, use_container_width=True)

        # --- Cumulative returns chart ---
        cum_chart = alt.Chart(df).mark_line(color="purple").encode(x="open_time:T", y="cumulative:Q").properties(title="Cumulative Strategy Returns")
        st.altair_chart(cum_chart, use_container_width=True)

        # --- Backtest summary ---
        st.subheader("Backtest Summary")
        for k, v in summary.items():
            if isinstance(v, float):
                st.write(f"**{k}:** {v:.2%}" if 'rate' in k.lower() or 'return' in k.lower() or 'drawdown' in k.lower() else f"**{k}:** {v}")
            else:
                st.write(f"**{k}:** {v}")
