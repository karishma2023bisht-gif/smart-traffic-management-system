"""
API layer — runs the detector in a background thread, and runs a SEPARATE
signal-cycling thread that gives each lane a real turn at green, with
duration proportional to that lane's traffic density. Emergency vehicles
immediately hijack the signal regardless of the current cycle.

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(str(Path(__file__).resolve().parent.parent))
from detection.detect import TrafficDetector  # noqa: E402

app = FastAPI(title="Smart Traffic Management API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VIDEO_SOURCE = "sample_videos/trafficsample1.mp4"  # change to 0 for webcam

LOW_THRESHOLD = 5
HIGH_THRESHOLD = 12

MIN_GREEN_SECONDS = 8       # every lane gets AT LEAST this much green time, even if empty
CYCLE_TOTAL_SECONDS = 60    # total time distributed across all lanes per full rotation
EMERGENCY_GREEN_SECONDS = 15  # how long an emergency vehicle holds the green

detector = TrafficDetector()
_lock = threading.Lock()
history = deque(maxlen=120)

signal_state = {
    "current_lane": None,
    "time_remaining": 0,
    "lane_order": [],
    "green_durations": {},
    "emergency_active": False,
}
_signal_lock = threading.Lock()


def classify_congestion(vehicle_count: int) -> str:
    if vehicle_count <= LOW_THRESHOLD:
        return "Low"
    elif vehicle_count <= HIGH_THRESHOLD:
        return "Medium"
    else:
        return "High"


def compute_green_durations(counts_by_lane: dict) -> dict:
    """Busier lanes get more seconds; every lane gets at least MIN_GREEN_SECONDS."""
    lanes = list(counts_by_lane.keys())
    if not lanes:
        return {}
    total = sum(counts_by_lane.values())
    if total == 0:
        equal_share = CYCLE_TOTAL_SECONDS // len(lanes)
        return {lane: max(MIN_GREEN_SECONDS, equal_share) for lane in lanes}
    return {
        lane: max(MIN_GREEN_SECONDS, round((count / total) * CYCLE_TOTAL_SECONDS))
        for lane, count in counts_by_lane.items()
    }


def camera_loop():
    """Continuously reads frames and updates detector stats."""
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    last_logged = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            cap = cv2.VideoCapture(VIDEO_SOURCE)
            continue

        with _lock:
            _, stats = detector.process_frame(frame)

        now = time.time()
        if now - last_logged >= 2:
            count = stats["total_vehicles_now"]
            history.append({
                "timestamp": now,
                "total_vehicles_now": count,
                "congestion_level": classify_congestion(count),
            })
            last_logged = now


def signal_loop():
    """
    Runs independently, once per second, giving each lane a real turn at
    green light, proportional to its density. Emergency vehicles interrupt
    the cycle immediately and hold the green until they've passed.
    """
    while True:
        time.sleep(1)

        with _lock:
            stats = detector.get_stats()
        counts_by_lane = stats.get("counts_by_lane", {})
        emergency_vehicles = stats.get("emergency_vehicles", [])

        if not counts_by_lane:
            continue

        with _signal_lock:
            # First run: set up the initial rotation
            if not signal_state["lane_order"]:
                signal_state["lane_order"] = list(counts_by_lane.keys())
                signal_state["green_durations"] = compute_green_durations(counts_by_lane)
                signal_state["current_lane"] = signal_state["lane_order"][0]
                signal_state["time_remaining"] = signal_state["green_durations"][signal_state["current_lane"]]

            # Emergency vehicle: interrupt immediately, hold green for it
            if emergency_vehicles:
                emergency_lane = emergency_vehicles[0]["lane"]
                if not signal_state["emergency_active"] or signal_state["current_lane"] != emergency_lane:
                    signal_state["current_lane"] = emergency_lane
                    signal_state["time_remaining"] = EMERGENCY_GREEN_SECONDS
                    signal_state["emergency_active"] = True
                else:
                    signal_state["time_remaining"] -= 1
                continue
            else:
                signal_state["emergency_active"] = False

            # Normal rotation: count down, then move to the next lane in order
            signal_state["time_remaining"] -= 1
            if signal_state["time_remaining"] <= 0:
                current_index = signal_state["lane_order"].index(signal_state["current_lane"])
                next_index = (current_index + 1) % len(signal_state["lane_order"])
                signal_state["current_lane"] = signal_state["lane_order"][next_index]
                signal_state["green_durations"] = compute_green_durations(counts_by_lane)
                signal_state["time_remaining"] = signal_state["green_durations"][signal_state["current_lane"]]


@app.on_event("startup")
def start_background_threads():
    threading.Thread(target=camera_loop, daemon=True).start()
    threading.Thread(target=signal_loop, daemon=True).start()


@app.get("/")
def root():
    return {"status": "Smart Traffic API is running"}


@app.get("/live-count")
def live_count():
    with _lock:
        stats = detector.get_stats()
    stats["congestion_level"] = classify_congestion(stats["total_vehicles_now"])

    with _signal_lock:
        stats["signal_state"] = dict(signal_state)

    return stats


@app.get("/history")
def get_history():
    return list(history)


@app.get("/health")
def health():
    return {"status": "ok"}