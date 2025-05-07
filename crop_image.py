def crop_image(image, coordinates):
    x1, y1, x2, y2 = coordinates
    if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
        return None
    return image[y1:y2, x1:x2]
