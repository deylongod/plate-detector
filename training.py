from ultralytics import YOLO

model = YOLO('./best.pt')

model.train(
    data='./dataset/data.yaml',
    epochs=30,
    imgsz=640,
    save_period=10,
    project='./models',
    name='trained_model'
)
