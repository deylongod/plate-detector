from ultralytics import YOLO

model = YOLO('./best.pt')

model.train(
    data='C:/Users/deylon/Desktop/numdetect/dataset/data.yaml',
    epochs=10,
    imgsz=640,
    save_period=10,
    project='C:/Users/deylon/Desktop/numdetect/saved_models',
    name='yolov8n_custom'
)