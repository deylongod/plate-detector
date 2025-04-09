import cv2
import easyocr
import re


def error_correction(text):
    letter_to_digit = {'З': '3', 'О': '0', 'Б': '6', 'Ь': '6', 'К': '1'}
    digit_to_letter = {'4': 'А', '5': 'Е'}
    letter_positions = {0, 4, 5}
    digit_positions = {1, 2, 3, 6, 7, 8}
    fixed_text = []
    for position, char in enumerate(text.upper()):
        if position in letter_positions:
            fixed_char = digit_to_letter.get(char, char)
        elif position in digit_positions:
            fixed_char = letter_to_digit.get(char, char)
        else:
            fixed_char = char
        fixed_text.append(fixed_char)
    return ''.join(fixed_text)


def image_preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    blurred = cv2.GaussianBlur(contrast, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def plate_validation(text):
    template = r"^([АВЕКМНОРСТУХ])(\d{3})([АВЕКМНОРСТУХ]{2})(\d{2,3})$"
    if match := re.match(template, text):
        return f"{match.group(1)}{match.group(2)}{match.group(3)}{match.group(4)}"
    return None


def process_license_plate(image):
    processed = image_preprocess(image)
    reader = easyocr.Reader(['ru'])
    result = reader.readtext(processed, allowlist="АВЕКМНОРСТУХ0123456789", detail=0)
    text = ''.join(result).upper()
    if not (formatted_text := plate_validation(text)):
        corrected_text = error_correction(text)
        formatted_text = plate_validation(corrected_text) or corrected_text
    return formatted_text, processed