def crop_image(image, coordinates):
    x1, y1, x2, y2 = coordinates
    return image[y1:y2, x1:x2]