import cv2
import easyocr
import re


def error_correction(text):
    letter_to_digit = {
        'З': '3', 'О': '0', 'Б': '6', 'Ь': '6', 'К': '1',
        'Ч': '4', 'Д': '5', 'Т': '7', 'В': '8', 'Я': '9',
        'A': '4', 'C': '0', 'E': '6', 'I': '1'
    }
    digit_to_letter = {
        '4': 'А', '5': 'Е', '0': 'О', '3': 'З',
        '7': 'Т', '8': 'В', '9': 'Р', '2': 'К'
    }
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
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    blurred = cv2.GaussianBlur(contrast, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def plate_validation(text):
    template = r"^([АВЕКМНОРСТУХ])(\d{3})([АВЕКМНОРСТУХ]{2})(\d{2,3})$"
    if match := re.match(template, text):
        return f"{match.group(1)}{match.group(2)}{match.group(3)}{match.group(4)}"
    return None


def scale_image(image, scale_factor):
    height, width = image.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)


def multi_scale_processing(image):
    scales = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    return [scale_image(image, scale) for scale in scales]


def select_best_result(results):
    valid_results = [r for r in results if r and 8 <= len(r) <= 9 and plate_validation(r)]
    if valid_results:
        return max(set(valid_results), key=valid_results.count)
    return max(results, key=len) if results else None


def process_license_plate(image):
    scaled_images = multi_scale_processing(image)
    reader = easyocr.Reader(['ru'])
    all_results = []
    processed_image = None
    for scaled_img in scaled_images:
        processed = image_preprocess(scaled_img)
        processed_image = processed
        result = reader.readtext(processed, allowlist="АВЕКМНОРСТУХ0123456789", detail=0)
        text = ''.join(result).upper()
        corrected = error_correction(text)
        validated = plate_validation(corrected) or corrected
        all_results.append(validated)
    best_result = select_best_result(all_results)
    return best_result, processed_image
