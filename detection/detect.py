"""
Core vehicle detection + tracking module, with lane assignment.

Divides the frame into NUM_LANES equal vertical strips and assigns each
detected vehicle to a lane based on where its center point falls.

Run with a video file:
    python detection/detect.py --source sample_videos/traffic1.mp4 --display

Run with your webcam:
    python detection/detect.py --source 0 --display
"""

import argparse
import time
from collections import defaultdict

import cv2
import supervision as sv
from ultralytics import YOLO

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
NUM_LANES = 3  # change this if your video shows more or fewer lanes


class TrafficDetector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.35, num_lanes=NUM_LANES):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.num_lanes = num_lanes

        self.latest_stats = {
            "total_vehicles_now": 0,
            "counts_by_type": defaultdict(int),
            "counts_by_lane": {f"lane_{i+1}": 0 for i in range(num_lanes)},
            "unique_vehicles_seen": set(),
            "last_updated": None,
        }

    def _assign_lane(self, center_x, frame_width):
        """Returns 'lane_1', 'lane_2', etc. based on horizontal position."""
        lane_width = frame_width / self.num_lanes
        lane_index = min(int(center_x // lane_width), self.num_lanes - 1)
        return f"lane_{lane_index + 1}"

    def process_frame(self, frame):
        frame_height, frame_width = frame.shape[:2]

        results = self.model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        mask = [
            (cls_id in VEHICLE_CLASSES) and (conf >= self.conf_threshold)
            for cls_id, conf in zip(detections.class_id, detections.confidence)
        ]
        detections = detections[mask] if len(mask) else detections
        detections = self.tracker.update_with_detections(detections)

        counts_by_type = defaultdict(int)
        counts_by_lane = {f"lane_{i+1}": 0 for i in range(self.num_lanes)}

        for box, cls_id, tracker_id in zip(detections.xyxy, detections.class_id, detections.tracker_id):
            label = VEHICLE_CLASSES.get(cls_id, "vehicle")
            counts_by_type[label] += 1
            self.latest_stats["unique_vehicles_seen"].add(tracker_id)

            center_x = (box[0] + box[2]) / 2
            lane = self._assign_lane(center_x, frame_width)
            counts_by_lane[lane] += 1

        self.latest_stats["total_vehicles_now"] = len(detections)
        self.latest_stats["counts_by_type"] = dict(counts_by_type)
        self.latest_stats["counts_by_lane"] = counts_by_lane
        self.latest_stats["last_updated"] = time.time()

        labels = [
            f"{VEHICLE_CLASSES.get(cls_id, 'vehicle')} #{tid}"
            for cls_id, tid in zip(detections.class_id, detections.tracker_id)
        ]
        annotated = self.box_annotator.annotate(frame.copy(), detections)
        annotated = self.label_annotator.annotate(annotated, detections, labels)

        # Draw lane divider lines for visual reference
        lane_width = frame_width / self.num_lanes
        for i in range(1, self.num_lanes):
            x = int(i * lane_width)
            cv2.line(annotated, (x, 0), (x, frame_height), (0, 255, 255), 2)

        return annotated, self.latest_stats

    def get_stats(self):
        stats = dict(self.latest_stats)
        stats["unique_vehicles_total"] = len(self.latest_stats["unique_vehicles_seen"])
        stats.pop("unique_vehicles_seen", None)
        return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Video file path or camera index")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model weights")
    parser.add_argument("--display", action="store_true", help="Show live window")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    detector = TrafficDetector()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        annotated, stats = detector.process_frame(frame)
        print(
            f"Vehicles: {stats['total_vehicles_now']} | Lanes: {stats['counts_by_lane']}",
            end="\r",
        )

        if args.display:
            cv2.imshow("Smart Traffic Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()