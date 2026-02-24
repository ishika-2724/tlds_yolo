import gradio as gr
import cv2
import tempfile
from ultralytics import YOLO

def yolov8_inference(image, video, model_path, image_size, conf_threshold):
    model = YOLO(r"C:\Users\ishik\Downloads\new best (1).pt")                # Load your trained YOLOv8 model

    if image:
        results = model(source=image, imgsz=image_size, conf=conf_threshold)
        annotated_image = results[0].plot()
        return annotated_image[:, :, ::-1], None   # Convert RGB to BGR for OpenCV display

    else:
        video_path = tempfile.mktemp(suffix=".mp4")
        with open(video_path, "wb") as f:
            with open(video, "rb") as g:
                f.write(g.read())

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_video_path = tempfile.mktemp(suffix=".mp4")
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(source=frame, imgsz=image_size, conf=conf_threshold)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        cap.release()
        out.release()

        return None, output_video_path

def app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Image", visible=True)
                video = gr.Video(label="Video", visible=False)
                input_type = gr.Radio(choices=["Image", "Video"], value="Image", label="Input Type")
                model_path = gr.Textbox(label="YOLOv8 Model Path or ID", value="yolov8n.pt")
                image_size = gr.Slider(label="Image Size", minimum=320, maximum=1280, step=32, value=640)
                conf_threshold = gr.Slider(label="Confidence Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.25)
                yolov8_infer = gr.Button(value="Detect Objects")

            with gr.Column():
                output_image = gr.Image(type="numpy", label="Annotated Image", visible=True)
                output_video = gr.Video(label="Annotated Video", visible=False)

        def update_visibility(input_type):
            return (gr.update(visible=input_type == "Image"),
                    gr.update(visible=input_type == "Video"),
                    gr.update(visible=input_type == "Image"),
                    gr.update(visible=input_type == "Video"))

        input_type.change(fn=update_visibility,
                          inputs=[input_type],
                          outputs=[image, video, output_image, output_video])

        def run_inference(image, video, model_path, image_size, conf_threshold, input_type):
            if input_type == "Image":
                return yolov8_inference(image, None, model_path, image_size, conf_threshold)
            else:
                return yolov8_inference(None, video, model_path, image_size, conf_threshold)

        yolov8_infer.click(fn=run_inference,
                           inputs=[image, video, model_path, image_size, conf_threshold, input_type],
                           outputs=[output_image, output_video])

gradio_app = gr.Blocks()
with gradio_app:
    gr.Markdown("# YOLOv8 Real-Time Object Detection Demo")
    app()

if __name__ == "__main__":
    gradio_app.launch()
