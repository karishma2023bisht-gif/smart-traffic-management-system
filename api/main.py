"""
API layer — runs the detector in a background thread, classifies congestion
level per lane, recommends signal priority for the busiest lane, keeps a
rolling history for trend charts, and exposes it all over HTTP.

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

detector = TrafficDetector()
_lock = threading.Lock()
history = deque(maxlen=120)


def classify_congestion(vehicle_count: int) -> str:
    if vehicle_count <= LOW_THRESHOLD:
        return "Low"
    elif vehicle_count <= HIGH_THRESHOLD:
        return "Medium"
    else:
        return "High"


def recommend_signal_priority(counts_by_lane: dict) -> dict:
    """
    Simple rule: the lane with the most vehicles gets recommended extra
    green time, proportional to its share of total traffic across lanes.
    """
    total = sum(counts_by_lane.values())
    if total == 0:
        return {"priority_lane": None, "reason": "No traffic detected", "green_time_seconds": {}}

    busiest_lane = max(counts_by_lane, key=counts_by_lane.get)

    # Distribute a 90-second signal cycle proportionally to each lane's share
    cycle_seconds = 90
    green_time_seconds = {
        lane: round((count / total) * cycle_seconds)
        for lane, count in counts_by_lane.items()
    }

    return {
        "priority_lane": busiest_lane,
        "reason": f"{busiest_lane} has the highest vehicle count ({counts_by_lane[busiest_lane]})",
        "green_time_seconds": green_time_seconds,
    }


def camera_loop():
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


@app.on_event("startup")
def start_background_capture():
    thread = threading.Thread(target=camera_loop, daemon=True)
    thread.start()


@app.get("/")
def root():
    return {"status": "Smart Traffic API is running"}


@app.get("/live-count")
def live_count():
    with _lock:
        stats = detector.get_stats()

    stats["congestion_level"] = classify_congestion(stats["total_vehicles_now"])
    stats["signal_recommendation"] = recommend_signal_priority(stats.get("counts_by_lane", {}))
    return stats


@app.get("/history")
def get_history():
    return list(history)


@app.get("/health")
def health():
    return {"status": "ok"}
