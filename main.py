import cv2
import glob
from detect_plates import detect_license_plate
from crop_image import crop_image
from recognize_text import process_license_plate


def display_image(image):
    resized_image = cv2.resize(image, None, fx=0.5, fy=0.5)
    cv2.imshow("plate-detector", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    model_path = "./best.pt"
    input_images = glob.glob("./tests/*.jpg") + glob.glob("./tests/*.jpeg") + glob.glob("./tests/*.png")
    print(f"Найдено изображений: {len(input_images)}")
    for image_path in input_images:
        image, plates = detect_license_plate(image_path, model_path)
        display_image(image)
        for i, plate in enumerate(plates):
            x1, y1, x2, y2 = plate  
            display_image(cv2.rectangle(image.copy(), (x1, y1), (x2, y2), (0, 255, 0), 2))
            cropped_image = crop_image(image, (x1, y1, x2, y2))
            display_image(cropped_image)
            plate_text, processed = process_license_plate(cropped_image)
            display_image(processed)
            print(f"Номер машины: [{plate_text}]")
            
            
if __name__ == "__main__":
    main()
