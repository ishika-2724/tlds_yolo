from ultralytics import YOLO

def train_model():
    # Load a base YOLOv8 model - you can choose yolov8n.pt, yolov8s.pt, etc. or your existing weights
    model = YOLO("yolov8n.pt") 

    # Setup training parameters
    model.train(
        data=r"C:\Users\ishik\Downloads\tlsdm project.v2i.yolov8\data.yaml",  # path to your dataset YAML file
        epochs=50,              
        imgsz=640,              # input image size, adjust to your GPU capacity
        batch=16,               # batch size, adjust based on your GPU memory
        lr0=0.001,              # initial learning rate
        device=0                # use GPU id 0, change if needed. Use 'cpu' for CPU
    )

if __name__ == "__main__":
    train_model()
