import cv2
import easyocr


def recognize_license_plate(image_path):
    image = cv2.imread(image_path)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    blurred = cv2.GaussianBlur(contrast, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow("ALPR", thresh)
    cv2.waitKey(0)
    
    reader = easyocr.Reader(['ru'])
    result = reader.readtext(thresh, allowlist="АВЕКМНОРСТУХ0123456789", detail=0)
    recognized_text = ''.join(result)
    
    return recognized_text


if __name__ == "__main__":
    image_path = "./cropped_images/C515HC142_plate.jpg"
    text = recognize_license_plate(image_path)
    print("Номер машины:", f"[{text}]")