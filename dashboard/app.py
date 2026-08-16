"""
Live dashboard — polls the FastAPI backend and displays vehicle counts,
congestion level, per-lane density, a live rotating traffic light with
countdown, emergency alerts, and a trend chart.

Run with:
    streamlit run dashboard/app.py
"""

import time

import pandas as pd
import requests
import streamlit as st

API_URL = "http://localhost:8000/live-count"
HISTORY_URL = "http://localhost:8000/history"
REFRESH_SECONDS = 1

st.set_page_config(page_title="Smart Traffic Dashboard", layout="wide")
st.title("🚦 Smart Traffic Management Dashboard")

placeholder = st.empty()

CONGESTION_COLORS = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}


def render_traffic_light(lane_name: str, is_green: bool, seconds_left: int, green_duration: int) -> str:
    red_on = "#3a3a3a" if is_green else "#e63946"
    green_on = "#2ecc71" if is_green else "#3a3a3a"
    countdown_text = f"{seconds_left}s" if is_green else f"waits {green_duration}s"

    return f"""
    <div style="display:flex; flex-direction:column; align-items:center; margin:10px;">
        <div style="font-weight:bold; margin-bottom:6px;">{lane_name.replace('_', ' ').title()}</div>
        <div style="background:#222; border-radius:12px; padding:10px; display:flex; flex-direction:column; gap:8px;">
            <div style="width:30px; height:30px; border-radius:50%; background:{red_on};"></div>
            <div style="width:30px; height:30px; border-radius:50%; background:{green_on};"></div>
        </div>
        <div style="margin-top:6px; font-size:0.85em; color:#aaa;">{countdown_text}</div>
    </div>
    """


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
            emergency_vehicles = data.get("emergency_vehicles", [])
            signal_state = data.get("signal_state", {})

            if emergency_vehicles:
                ev = emergency_vehicles[0]
                st.error(f"🚨 EMERGENCY VEHICLE DETECTED — {ev['type'].upper()} in {ev['lane'].replace('_', ' ').title()} — SIGNAL OVERRIDE ACTIVE 🚨")

            level = data.get("congestion_level", "Low")
            emoji = CONGESTION_COLORS.get(level, "🟢")

            if not emergency_vehicles:
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

            # --- Live rotating traffic light ---
            st.subheader("🚥 Live Signal Status (auto-rotating by density)")
            current_lane = signal_state.get("current_lane")
            time_remaining = signal_state.get("time_remaining", 0)
            green_durations = signal_state.get("green_durations", {})

            if green_durations:
                light_cols = st.columns(len(green_durations))
                for col, lane in zip(light_cols, green_durations.keys()):
                    is_green = (lane == current_lane)
                    col.markdown(
                        render_traffic_light(lane, is_green, time_remaining, green_durations.get(lane, 0)),
                        unsafe_allow_html=True,
                    )

                if signal_state.get("emergency_active"):
                    st.error(f"**🚨 Holding green for emergency vehicle in {current_lane.replace('_', ' ').title()} — {time_remaining}s remaining**")
                elif current_lane:
                    st.info(f"**{current_lane.replace('_', ' ').title()} has the green light for {time_remaining}s** — busier lanes get more time, quieter lanes wait a bit longer.")
            else:
                st.info("Waiting for traffic data to start the signal cycle...")

            st.subheader("🛣️ Traffic density by lane")
            lane_counts = data.get("counts_by_lane", {})
            if lane_counts:
                st.bar_chart(lane_counts)

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
    