import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from detect_plates import detect_license_plate
from crop_image import crop_image
from recognize_text import process_license_plate, plate_validation
import datetime
from deep_sort_realtime.deepsort_tracker import DeepSort


def put_cyrillic_text(img, text, pos):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("./fonts/russian.ttf", 20)
    draw.text(pos, text, font=font, fill=(0, 255, 0))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def process_video(input_path, output_path, plate_model_path, frame_skip=5):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Не удалось открыть видеофайл")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f"tests/video/results/{output_path}", fourcc, fps, (width, height))
    plate_tracker = DeepSort(max_age=50, n_init=3)
    logged_plates = {}
    frame_count = 0
    
    with open("tests/video/results/log.txt", "a") as log_file:
        log_file.write(f"\nВидеофайл: {input_path.split('/')[-1].split('\\')[-1]}\n")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_skip != 0:
            out.write(frame)
            continue
        try:
            _, plates = detect_license_plate(frame, plate_model_path)
            plate_detections = []
            for (x1, y1, x2, y2) in plates:
                if (x2 - x1) < 50 or (y2 - y1) < 20:
                    continue
                plate_detections.append(([x1, y1, x2-x1, y2-y1], 0.9, 'plate'))
            plate_tracks = plate_tracker.update_tracks(plate_detections, frame=frame)
            for track in plate_tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                cropped = crop_image(frame, (int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])))
                if cropped is None or cropped.size == 0:
                    continue
                plate_text, _ = process_license_plate(cropped)
                if plate_text and plate_validation(plate_text):
                    frame = put_cyrillic_text(frame, plate_text, (int(ltrb[0]), int(ltrb[1]) - 30))
                    cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
                    if track_id not in logged_plates:
                        now = datetime.datetime.now()
                        log_entry = f"CLS:{track_id} {plate_text} {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
                        with open("tests/video/results/log.txt", "a") as log_file:
                            log_file.write(log_entry)
                        logged_plates[track_id] = plate_text
        except Exception as e:
            print("Ошибка в кадре")
        out.write(frame)
    cap.release()
    out.release()
    print("Видео и логи сохранены в директории tests/video/results")