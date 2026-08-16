
import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()
rf = Roboflow(api_key=os.getenv("E3XNXTB7Y6rELIvZbWMN")) 

project = rf.workspace("yolo-fn1iu").project(
    "emergency-vehicles-detection-xockh-af7sr"
)

dataset = project.version(1).download(
    model_format="yolov8",
    location="./emergency-vehicles-yolov8"
)

print("Downloaded to:", dataset.location)