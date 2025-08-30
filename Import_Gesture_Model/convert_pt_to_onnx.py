from ultralytics import YOLO

model = YOLO("G:/ISDN3002/runs/detect/train6/weights/best.pt")
model.export(format="onnx", imgsz=640, dynamic=True, half=False, simplify=False, opset=12)
