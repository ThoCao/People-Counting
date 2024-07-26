import csv
import os
import cv2
import torch
import numpy as np
import random
from typing import Tuple

# Setup video writer if needed
def setup_video_writer(cfg, frame_width: int, frame_height: int, model_name: str) -> cv2.VideoWriter:
    output_video_path = f"{cfg.video_output.split('.')[0]}_{model_name}.mp4"
    return cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (frame_width, frame_height))

# Initialize CSV for writing results
def initialize_csv(filename: str) -> Tuple[any, csv.writer]:
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    file = open(filename, mode='w', newline='', encoding='utf-8')
    writer = csv.writer(file)
    writer.writerow(["Frame Name", "People Count"])
    return file, writer

# Resize image to be divisible by stride
def resize_image(img: np.ndarray, stride: int = 32) -> np.ndarray:
    h, w = img.shape[:2]
    new_h, new_w = (int(np.ceil(h / stride)) * stride, int(np.ceil(w / stride)) * stride)
    resized_img = cv2.resize(img, (new_w, new_h))
    return resized_img

# Draw bounding box and label on the frame
def draw_bounding_box(frame: np.ndarray, box, class_name: str, conf: float) -> None:
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    label = f"{class_name}: {conf:.2f}"
    text_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
    c2 = (x1 + text_size[0], y1 - text_size[1] - 3)
    cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
    cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

# Set random seed for reproducibility
def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Initialize the random seed
set_random_seed(42)