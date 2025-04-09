from ultralytics import YOLO
import cv2


def detect_license_plate(image_path, model_path):
    model = YOLO(model_path)
    image = cv2.imread(image_path)
    results = model.predict(source=image_path)
    plates = []
    for result in results:
        for box in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4])
            plates.append((x1, y1, x2, y2))
    return image, plates