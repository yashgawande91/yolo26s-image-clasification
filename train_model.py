from ultralytics import YOLO


model = YOLO("YOLO26s-cls")

result = model.train(data="custom_dataset", epochs=10, imgsz=640)