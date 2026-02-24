import cv2
from ultralytics import YOLO

def main():
    # Load your trained YOLOv8 model weights - update the path accordingly
    model = YOLO(r"C:\Users\ishik\Downloads\new best (1).pt")  # Replace 'best.pt' with your model path

    # Open webcam (0 is default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Run inference on the current frame with confidence threshold 0.25
        results = model(frame, conf=0.25)

        # Annotate the frame with bounding boxes and labels
        annotated_frame = results[0].plot()

        # Display the resulting frame
        cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
