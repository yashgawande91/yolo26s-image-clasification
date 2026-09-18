from ultralytics import YOLO


model = YOLO("runs/classify/train/weights/best.pt")


result = model("test_images/image2.jpg", save=True)
result[0].show()


