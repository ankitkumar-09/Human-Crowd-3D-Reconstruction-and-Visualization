from ultralytics import YOLO

# This line triggers the download of yolov8n.pt if it is not already present
model = YOLO("yolov8n.pt")

# Just to confirm it loaded
print("Model loaded:", model)
