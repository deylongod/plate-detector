import easyocr


def recognize_license_plate_easyocr(image_path):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image_path)
    return result


if __name__ == "__main__":
    image_path = "./tests/cropped_plate_0.jpg"
    recognized_text = recognize_license_plate_easyocr(image_path)
    for detection in recognized_text:
        bbox, text, _ = detection
        print(f"Номер машины: {text}")