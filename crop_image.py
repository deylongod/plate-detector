import cv2
from detect_plates import detect_license_plate


def crop_image(image_path, x1, y1, x2, y2, output_directory):
    image = cv2.imread(image_path)
    cropped_image = image[y1:y2, x1:x2]
    
    file_name = image_path.split('/')[-1].split('.')[0]
    output_file = f"{output_directory}/{file_name}_plate.jpg"
    cv2.imwrite(output_file, cropped_image)
    
    cv2.imshow("ALPR", cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "./tests/good__visibility/C515HC142.jpg"
    model_path = "./best.pt"
    output_directory = './cropped_images'
    
    _, plates = detect_license_plate(image_path, model_path)
    for i, plate in enumerate(plates):
        x1, y1, x2, y2 = plate
        crop_image(image_path, x1, y1, x2, y2, output_directory)

