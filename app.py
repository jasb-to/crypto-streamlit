import streamlit as st
import requests
import pandas as pd
import altair as alt

# Function to fetch data from Moralis
def fetch_moralis_data(address, chain, from_timestamp, to_timestamp, interval):
    url = f"https://api.moralis.io/v1/ohlcv"
    params = {
        "address": address,
        "chain": chain,
        "from": from_timestamp,
        "to": to_timestamp,
        "interval": interval
    }
    headers = {"X-API-Key": "YOUR_MORALIS_API_KEY"}
    response = requests.get(url, params=params, headers=headers)
    return response.json()

# Function to fetch data from Syve
def fetch_syve_data(token, chain, from_timestamp, to_timestamp, interval):
    url = f"https://api.syve.ai/v1/price/historical/ohlc"
    params = {
        "token": token,
        "chain": chain,
        "from": from_timestamp,
        "to": to_timestamp,
        "interval": interval
    }
    headers = {"Authorization": "Bearer YOUR_SYVE_API_KEY"}
    response = requests.get(url, params=params, headers=headers)
    return response.json()

# Streamlit UI
st.title("Crypto OHLCV Data Fetcher")

token = st.selectbox("Select Token", ["AI16Z", "ADA"])
chain = st.selectbox("Select Chain", ["ethereum", "polygon"])
from_timestamp = st.number_input("From Timestamp", min_value=0)
to_timestamp = st.number_input("To Timestamp", min_value=0)
interval = st.selectbox("Select Interval", ["1m", "5m", "1h", "1d"])

api_choice = st.radio("Choose API", ["Moralis", "Syve"])

if st.button("Fetch Data"):
    if api_choice == "Moralis":
        data = fetch_moralis_data(token, chain, from_timestamp, to_timestamp, interval)
    else:
        data = fetch_syve_data(token, chain, from_timestamp, to_timestamp, interval)

    if data:
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        chart = alt.Chart(df).mark_line().encode(
            x='timestamp:T',
            y='close:Q'
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.error("Failed to fetch data.")
