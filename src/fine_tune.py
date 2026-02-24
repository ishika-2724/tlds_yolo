from ultralytics import YOLO

def finetune_model():
    model = YOLO(r"C:\Users\ishik\Downloads\new best (1).pt")  # Load existing model weights

    model.train(
        data=r"C:\Users\ishik\Downloads\tlsdm project.v2i.yolov8\data.yaml",  # Path to dataset YAML
        epochs=40,             # Number of epochs
        imgsz=640,             # Image size
        batch=16,              # Batch size (note the comma here)
        lr0=0.001              # Learning rate
    )

if __name__ == "__main__":
    finetune_model()
