import os
import sys
import csv
import torch
import cv2
import numpy as np
from typing import List

import hydra
from omegaconf import DictConfig
from deep_sort_realtime.deepsort_tracker import DeepSort
from boxmot.trackers.botsort.bot_sort import BoTSORT
from pathlib import Path

from my_utils import (
    setup_video_writer, initialize_csv, resize_image, 
    set_random_seed
)

sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov10'))

# Initialize the random seed
set_random_seed(42)

def build_model(model_type: str, model_name: str):
    """
    Builds and loads a model based on the specified type and name.
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

def process_frame_yolov10(frame: np.ndarray, model, coco_class_names: List[str], device: torch.device) -> List[tuple]:
    """
    Processes a frame using the YOLOv10 model to detect objects.
    """
    original_h, original_w = frame.shape[:2]
    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).to(device)
    img = img.permute(2, 0, 1).unsqueeze(0).float()
    img /= 255.0

    results = model.predict(img, conf=0.25)
    detections = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls == 0:
                x1, y1, x2, y2 = box.xyxy[0]
                x1 = int(x1 * original_w / 640)
                y1 = int(y1 * original_h / 640)
                x2 = int(x2 * original_w / 640)
                y2 = int(y2 * original_h / 640)
                detections.append((x1, y1, x2, y2, box.conf[0], int(cls)))

    return detections

def process_frame_yolov5(frame: np.ndarray, model, coco_class_names: List[str], device: torch.device, non_max_suppression) -> List[tuple]:
    """
    Processes a frame using the YOLOv5 model to detect objects.
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
    
    detections = []
    for det in pred:
        if len(det):
            for *xyxy, conf, cls in reversed(det):
                if cls != 0:
                    continue
                x1, y1, x2, y2 = map(int, xyxy)
                detections.append((x1, y1, x2, y2, conf.item(), int(cls)))

    return detections

def process_frame(frame: np.ndarray, model, coco_class_names: List[str], model_type: str, device: torch.device = None, non_max_suppression=None) -> int:
    """
    Processes a single frame to detect objects based on the model type.
    """
    if model_type == "yolov10":
        return process_frame_yolov10(frame, model, coco_class_names, device)
    elif model_type == "yolov5":
        return process_frame_yolov5(frame, model, coco_class_names, device, non_max_suppression)
    else:
        raise ValueError("Unsupported model type")

def handle_frame_processing(cfg: DictConfig, cap: cv2.VideoCapture, writer: csv.writer, model, device: torch.device, non_max_suppression=None, out: cv2.VideoWriter = None) -> None:
    """
    Processes frames from a video capture, detecting objects, and writing results.
    """
    # Initialize BoTSORT tracker
    tracker_device = 'cuda:0' if torch.cuda.is_available() and device == 'cuda' else 'cpu'
    tracker_BoTSort = BoTSORT(
        model_weights= Path(cfg.tracker_model),
        device= tracker_device,  
        fp16= False,
        )
    unique_ids = set()
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame. Exiting.")
            break

        count += 1
        detections = process_frame(frame, model, cfg.coco_class_names, cfg.model_type, device, non_max_suppression)

        detections = np.array(detections)

        # Update tracker with detections
        tracks = tracker_BoTSort.update(detections, frame)
        # Draw bounding boxes and track IDs on the frame
        for track in tracks:
            x1, y1, x2, y2, track_id = track[:5]
            unique_ids.add(track_id)  # Add track_id to unique_ids set
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {int(track_id)}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
        # Display the count of unique IDs
        cv2.putText(frame, f"Unique People Count: {len(unique_ids)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        print(f"Frame Count: {count}, Unique People: {len(unique_ids)}")
        frame_name = f"frame_{count}"
        writer.writerow([frame_name, len(unique_ids)])

        if cfg.save_imgs:
            frame_path = os.path.join(cfg.debug_imgs, frame_name + ".png")
            cv2.imwrite(frame_path, frame)
            
        if cfg.save_video and out is not None:
            out.write(frame)

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
    """
    cfg.video_input = os.path.abspath(cfg.video_input)
    cfg.csv_output = os.path.abspath(cfg.csv_output)
    cfg.video_output = os.path.abspath(cfg.video_output)
    cfg.debug_imgs = os.path.abspath(cfg.debug_imgs)
    
    model, device, non_max_suppression = build_model(cfg.model_type, cfg.model)

    cap = cv2.VideoCapture(cfg.video_input)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {cfg.video_input}")
        return

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

