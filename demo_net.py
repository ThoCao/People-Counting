import time
import torch
import cv2
import os
import sys
import csv
from typing import List
import numpy as np

import hydra
from omegaconf import DictConfig

from my_utils import setup_video_writer, initialize_csv, resize_image, draw_bounding_box, set_random_seed

# Add paths to the sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov10'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))

# Initialize the random seed
set_random_seed(42)

def build_model(model_type: str, model_name: str):
    """
    Builds and loads a model based on the specified type and name.

    Args:
        model_type (str): The type of model (e.g., 'yolov10', 'yolov5').
        model_name (str): The name of the model to load.

    Returns:
        tuple: The loaded model, the device it is on, and a non-max suppression function (if applicable).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_type == "yolov10":
        from ultralytics import YOLOv10
        model = YOLOv10.from_pretrained(f"jameslahm/{model_name}")
        model.to(device)
        return model, device, None
    elif model_type == "yolov5":
        from yolov5.models.common import DetectMultiBackend
        from yolov5.utils.general import non_max_suppression
        model = DetectMultiBackend(f"{model_name}.pt", device=device)
        return model, device, non_max_suppression
    else:
        raise ValueError("Unsupported model type")

def process_frame_yolov10(frame: np.ndarray, model, coco_class_names: List[str], device: torch.device) -> int:
    """
    Processes a frame using the YOLOv10 model to detect objects.

    Args:
        frame (np.ndarray): The input frame to process.
        model: The YOLOv10 model.
        coco_class_names (List[str]): List of COCO class names.
        device (torch.device): The device to run the model on.

    Returns:
        int: The number of detected objects.
    """
    original_h, original_w = frame.shape[:2]
    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).to(device)
    img = img.permute(2, 0, 1).unsqueeze(0).float()
    img /= 255.0

    results = model.predict(img, conf=0.25)
    detected_objects = 0

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls < len(coco_class_names) and coco_class_names[cls] in coco_class_names:
                x1, y1, x2, y2 = box.xyxy[0]
                x1 = int(x1 * original_w / 640)
                y1 = int(y1 * original_h / 640)
                x2 = int(x2 * original_w / 640)
                y2 = int(y2 * original_h / 640)
                draw_bounding_box(frame, (x1, y1, x2, y2), coco_class_names[cls], box.conf[0])
                detected_objects += 1

    return detected_objects

def process_frame_yolov5(frame: np.ndarray, model, coco_class_names: List[str], device: torch.device, non_max_suppression) -> int:
    """
    Processes a frame using the YOLOv5 model to detect objects.

    Args:
        frame (np.ndarray): The input frame to process.
        model: The YOLOv5 model.
        coco_class_names (List[str]): List of COCO class names.
        device (torch.device): The device to run the model on.
        non_max_suppression: Function to apply non-max suppression on the model predictions.

    Returns:
        int: The number of detected objects.
    """
    img = resize_image(frame)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).to(device)
    img = img.permute(2, 0, 1).float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, 0.4, 0.5, classes=[0], agnostic=False)
    
    detected_objects = 0
    for det in pred:
        if len(det):
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = map(int, xyxy)
                draw_bounding_box(frame, (x1, y1, x2, y2), coco_class_names[int(cls)], conf)
                detected_objects += 1

    return detected_objects

def process_frame(frame: np.ndarray, model, coco_class_names: List[str], model_type: str, device: torch.device = None, non_max_suppression=None) -> int:
    """
    Processes a single frame to detect objects based on the model type.

    Args:
        frame (np.ndarray): The input frame to process.
        model: The model to use for detection.
        coco_class_names (List[str]): List of COCO class names.
        model_type (str): The type of model ('yolov10' or 'yolov5').
        device (torch.device, optional): The device to run the model on.
        non_max_suppression (optional): Non-max suppression function for YOLOv5.

    Returns:
        int: The number of detected objects.
    """
    if model_type == "yolov10":
        return process_frame_yolov10(frame, model, coco_class_names, device)
    elif model_type == "yolov5":
        return process_frame_yolov5(frame, model, coco_class_names, device, non_max_suppression)
    else:
        raise ValueError("Unsupported model type")

def handle_frame_processing(cfg: DictConfig, cap: cv2.VideoCapture, writer: csv.writer, model, device: torch.device, non_max_suppression=None, out: cv2.VideoWriter = None) -> None:
    """
    Processing frames from a video capture, detecting objects, and writing results.

    Args:
        cfg (DictConfig): Configuration object.
        cap (cv2.VideoCapture): Video capture object.
        writer (csv.writer): CSV writer object for writing results.
        model: The model to use for detection.
        device (torch.device): The device to run the model on.
        non_max_suppression (optional): Non-max suppression function for YOLOv5.
        out (cv2.VideoWriter, optional): Video writer object for saving output video.
    """
    count = 0
    ptime = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame. Exiting.")
            break

        count += 1
        detected_objects = process_frame(frame, model, cfg.coco_class_names, cfg.model_type, device, non_max_suppression)
        print(f"Frame Count: {count}")
        print(f"Detected Objects: {detected_objects}")

        frame_name = f"frame_{count}"
        writer.writerow([frame_name, detected_objects])

        # Draw the number of detected people on the image
        cv2.putText(frame, f"People Count: {detected_objects}", (10, 150), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        # Save debug image if enabled
        if cfg.save_imgs:
            frame_path = os.path.join(cfg.debug_imgs, frame_name + ".png")
            cv2.imwrite(frame_path, frame)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(frame, f"FPS: {str(int(fps))}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        # Write to output video if enabled
        if cfg.save_video and out is not None:
            out.write(frame)

        # Show image if enabled
        if cfg.image_show:
            cv2.imshow("frame detection", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                print("Exit signal received. Exiting.")
                break

        if count >= cfg.max_frames:
            print("Reached max frames. Exiting.")
            break

@hydra.main(version_base=None, config_path="./cfg", config_name="demo_config")
def main(cfg: DictConfig) -> None:
    """
    Main function to set up the configuration and execute the frame processing.

    Args:
        cfg (DictConfig): Configuration object.
    """
    # Ensure all paths are correctly expanded
    cfg.video_input = os.path.abspath(cfg.video_input)
    cfg.csv_output = os.path.abspath(cfg.csv_output)
    cfg.video_output = os.path.abspath(cfg.video_output)
    cfg.debug_imgs = os.path.abspath(cfg.debug_imgs)
    
    model, device, non_max_suppression = build_model(cfg.model_type, cfg.model)

    cap = cv2.VideoCapture(cfg.video_input)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {cfg.video_input}")
        return

    # Create directory for the CSV file if it doesn't exist
    csv_dir = os.path.dirname(cfg.csv_output)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    csv_filename = f"{cfg.csv_output.split('.')[0]}_{cfg.model}.csv"
    file, writer = initialize_csv(csv_filename)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not os.path.exists(cfg.debug_imgs):
        os.makedirs(cfg.debug_imgs)

    out = setup_video_writer(cfg, frame_width, frame_height, cfg.model) if cfg.save_video else None

    handle_frame_processing(cfg, cap, writer, model, device, non_max_suppression, out)

    cap.release()
    
    if out is not None:
        out.release()

    cv2.destroyAllWindows()
    file.close()

if __name__ == "__main__":
    main()
