"""
Live dashboard — polls the FastAPI backend and displays vehicle counts,
congestion level, per-lane density, signal recommendations, and a trend chart.

Run with:
    streamlit run dashboard/app.py
"""

import time

import pandas as pd
import requests
import streamlit as st

API_URL = "http://localhost:8000/live-count"
HISTORY_URL = "http://localhost:8000/history"
REFRESH_SECONDS = 2

st.set_page_config(page_title="Smart Traffic Dashboard", layout="wide")
st.title("🚦 Smart Traffic Management Dashboard")

placeholder = st.empty()

CONGESTION_COLORS = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}

while True:
    try:
        data = requests.get(API_URL, timeout=3).json()
        history = requests.get(HISTORY_URL, timeout=3).json()
    except requests.exceptions.RequestException:
        data, history = None, None

    with placeholder.container():
        if data is None:
            st.error("Could not reach the API. Is it running on localhost:8000?")
        else:
            level = data.get("congestion_level", "Low")
            emoji = CONGESTION_COLORS.get(level, "🟢")

            if level == "High":
                st.error(f"{emoji} HIGH CONGESTION — {data.get('total_vehicles_now', 0)} vehicles in frame")
            elif level == "Medium":
                st.warning(f"{emoji} Moderate traffic — {data.get('total_vehicles_now', 0)} vehicles in frame")
            else:
                st.success(f"{emoji} Traffic flowing normally — {data.get('total_vehicles_now', 0)} vehicles in frame")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Vehicles now", data.get("total_vehicles_now", 0))
            col2.metric("Unique vehicles seen", data.get("unique_vehicles_total", 0))
            col3.metric("Congestion level", f"{emoji} {level}")
            col4.metric("Last updated", time.strftime("%H:%M:%S"))

            st.subheader("🛣️ Traffic density by lane")
            lane_counts = data.get("counts_by_lane", {})
            if lane_counts:
                st.bar_chart(lane_counts)

            st.subheader("🚦 Signal timing recommendation")
            recommendation = data.get("signal_recommendation", {})
            priority_lane = recommendation.get("priority_lane")
            if priority_lane:
                st.info(f"**Priority lane: {priority_lane}** — {recommendation.get('reason', '')}")
                green_times = recommendation.get("green_time_seconds", {})
                cols = st.columns(len(green_times)) if green_times else []
                for col, (lane, seconds) in zip(cols, green_times.items()):
                    col.metric(f"{lane} green time", f"{seconds}s")
            else:
                st.info("Not enough traffic data yet to recommend signal timing.")

            st.subheader("Breakdown by vehicle type")
            counts = data.get("counts_by_type", {})
            if counts:
                st.bar_chart(counts)

            st.subheader("Traffic trend (last few minutes)")
            if history:
                df = pd.DataFrame(history)
                df["time"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime("%H:%M:%S")
                st.line_chart(df.set_index("time")["total_vehicles_now"])
            else:
                st.info("Collecting trend data... check back in a few seconds.")

    time.sleep(REFRESH_SECONDS)