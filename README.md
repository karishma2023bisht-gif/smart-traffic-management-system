# Smart Traffic Management System

Real-time vehicle detection and traffic monitoring system built with YOLOv8 and OpenCV. Detects and tracks cars, trucks, buses, and motorcycles from live camera or video feeds, serves live counts through a FastAPI backend, and displays them on a Streamlit dashboard. Designed to run on Raspberry Pi. 

## Features

- Real-time vehicle detection using YOLOv8 (car, truck, bus, motorcycle)
- Vehicle tracking with persistent IDs to avoid double-counting
- Lane-based traffic density detection with signal timing recommendations
- Emergency vehicle detection (ambulance, fire truck) with signal priority override
- FastAPI backend serving live vehicle counts as JSON
- Streamlit dashboard for live visualization, congestion alerts, and trend charts
- Designed for deployment on Raspberry Pi with a live camera feed

## Project structure
smart-traffic-system/
├── requirements.txt # Python dependencies
├── detection/
│ └── detect.py # YOLO + OpenCV detection, tracking & lane assignment
├── api/
│ └── main.py # FastAPI backend, serves live stats as JSON
├── dashboard/
│ └── app.py # Streamlit dashboard, polls the API
├── download_dataset.py # Downloads the emergency vehicle dataset
├── train_emergency_model.py # Trains the custom emergency vehicle model
└── sample_videos/ # Test traffic footage (not tracked in Git)
## Setup

```bash
python -m venv venv
venv\Scripts\activate        # on Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## Running the project

**1. Test detection standalone:**
```bash
python detection/detect.py --source sample_videos/yourvideo.mp4 --display
```

**2. Run the API (in one terminal):**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**3. Run the dashboard (in a second terminal):**
```bash
streamlit run dashboard/app.py
```

Then open `http://localhost:8501` to see the live dashboard.

## Notes

- Vehicle classes detected are the standard COCO classes YOLOv8 already knows — no custom training needed for basic detection.
- Tracking is handled by `supervision`'s ByteTrack, assigning a persistent ID to each vehicle.
- Emergency vehicle detection uses a custom-trained YOLOv8 model, fine-tuned on a labeled ambulance/fire truck dataset.
- For Raspberry Pi deployment, switch `VIDEO_SOURCE` in `api/main.py` to `0` for a live camera feed.
