from ultralytics import YOLO

# Load pretrained YOLOv8 model (you can use 'yolov8n.pt', 'yolov8s.pt', etc.)
model = YOLO("yolov8n.pt")

# Train the model
model.train(data="data.yaml", epochs=50, imgsz=640, batch=16)
