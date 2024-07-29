import os
import random
from tqdm import tqdm
import zipfile
from pycocotools.coco import COCO
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
from hydra import initialize, compose
import json
import gdown
from collections import defaultdict
import numpy as np
from omegaconf import OmegaConf

from data_manager import DatasetManager
from my_utils import verify_labels, draw_boxes

def main(cfg: Dict[str, Any]) -> None:
    """
    Main function to process and verify dataset images.

    Args:
        cfg (Dict[str, Any]): Configuration dictionary containing dataset settings.
    """
    dataset_cfg = cfg.dataset  # Access the nested dataset configuration
    
    # Initialize DatasetManager with the configuration
    dataset_manager = DatasetManager(dataset_cfg)
    img_ids = dataset_manager.get_img_ids(category_name='person', sample_size=cfg.sample_size)
    print(f"Sampled image IDs: {img_ids}")
    
    if not img_ids:
        print("No images available for sampling.")
        return

    dataset = dataset_manager.create_dataset(img_ids, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    # Save labels in YOLO format
    dataset_manager.save_labels(img_ids)

    # Verify labels by displaying images with bounding boxes
    verify_labels(dataset_cfg)

    for idx, (images, targets) in enumerate(dataloader):
        if idx >= cfg.display_limit:
            break

        # Convert the tensor image to PIL image
        image = transforms.ToPILImage()(images[0])
        boxes = targets['boxes'].tolist()

        # Print debug information
        print(f"Image {idx + 1} has {len(boxes)} bounding boxes.")
        for box in boxes:
            print(f"Bounding box: {box}")

        # Draw the boxes on the image
        image_with_boxes = draw_boxes(image, boxes)

        plt.imshow(image_with_boxes)
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    with initialize(config_path="config"):
        cfg = compose(config_name="config")
        print(OmegaConf.to_yaml(cfg))
        main(cfg)
