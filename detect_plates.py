from ultralytics import YOLO
import cv2


def detect_license_plate(image_input, model_path):
    model = YOLO(model_path)
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
    else:
        image = image_input.copy()
    results = model.predict(source=image)
    detections = []
    for result in results:
        for box in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4])
            detections.append((x1, y1, x2, y2))
    return image, detections

