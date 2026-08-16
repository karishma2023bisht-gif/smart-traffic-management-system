from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data="emergency-vehicles-yolov8/data.yaml",
        epochs=15,
        imgsz=416,
        batch=8,
        name="emergency_vehicle_model",
    )

if __name__ == "__main__":
    main()
