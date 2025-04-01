from ultralytics import YOLO
import cv2


def detect_license_plate(image_path, model_path):
    model = YOLO(model_path)
    image = cv2.imread(image_path)
    results = model(image_path)
    plates = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plates.append(x1, y1, x2, y2)
    return image, plates


image_path = './tests/good__visibility/C515HC142.jpg'
# model_path = './saved_models/yolov8n_custom/weights/epoch0.pt'
model_path = './best.pt'

image, plates = detect_license_plate(image_path, model_path)

for plate in plates:
    x1, y1, x2, y2 = plate
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

resized__image = cv2.resize(image, None, fx=0.5, fy=0.5)

cv2.imshow("ALPR", resized__image)
cv2.waitKey(0)
cv2.destroyAllWindows()