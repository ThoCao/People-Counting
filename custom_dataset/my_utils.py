import os
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def draw_boxes(image: Image.Image, boxes: List[Tuple[int, int, int, int]]) -> Image.Image:
    """
    Draws bounding boxes on the given image.

    Args:
        image (Image.Image): The image on which to draw the bounding boxes.
        boxes (List[Tuple[int, int, int, int]]): A list of bounding boxes, each represented by a tuple (xmin, ymin, width, height).

    Returns:
        Image.Image: The image with bounding boxes drawn on it.
    """
    draw = ImageDraw.Draw(image)
    for box in boxes:
        if len(box) == 0:  # Skip empty boxes
            continue
        if isinstance(box[0], list):  # Handling nested lists
            for sub_box in box:
                xmin, ymin, width, height = sub_box
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmin + width), int(ymin + height)
                draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=2)
        else:
            xmin, ymin, width, height = box
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmin + width), int(ymin + height)
            draw.rectangle([xmin, ymin, xmax, ymax], outline='red', width=2)
    return image

def verify_labels(cfg: Dict[str, Any], display_limit: int = 10) -> None:
    """
    Verifies the labels by drawing bounding boxes on images and displaying them.

    Args:
        cfg (Dict[str, Any]): Configuration dictionary containing dataset settings.
        display_limit (int): Number of images to display for verification. Default is 10.
    """
    dataset_cfg = cfg['dataset']  # Access the nested dataset configuration
    data_dir = dataset_cfg['data_dir']
    image_dir = dataset_cfg['image_dir']
    label_dir = os.path.join(data_dir, 'labels')

    img_ids = [img_id for img_id in os.listdir(image_dir) if img_id.endswith('.jpg')]
    label_ids = [os.path.splitext(label_id)[0] for label_id in os.listdir(label_dir) if label_id.endswith('.txt')]

    # Filter to keep only those with the same name
    img_ids_filtered = [img_id for img_id in img_ids if os.path.splitext(img_id)[0] in label_ids]
    img_paths = [os.path.join(image_dir, img_id) for img_id in img_ids_filtered]
    label_paths = [os.path.join(label_dir, os.path.splitext(img_id)[0] + '.txt') for img_id in img_ids_filtered]

    display_count = 0

    for img_path, label_path in zip(img_paths, label_paths):
        if display_count >= display_limit:
            break

        # Read image
        img = Image.open(img_path)
        img_width, img_height = img.size

        # Read label file
        if not os.path.exists(label_path):
            print(f"Label file {label_path} does not exist.")
            continue

        with open(label_path, 'r') as f:
            lines = f.readlines()

        boxes = []
        for line in lines:
            # Parse YOLO format: class x_center y_center width height
            parts = line.strip().split()
            class_id, x_center, y_center, width, height = map(float, parts)

            # Convert from YOLO format to bounding box format
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height

            xmin = int(x_center - width / 2)
            ymin = int(y_center - height / 2)
            xmax = int(x_center + width / 2)
            ymax = int(y_center + height / 2)

            boxes.append([xmin, ymin, xmax, ymax])

        # Draw bounding boxes on image
        draw = ImageDraw.Draw(img)
        for box in boxes:
            draw.rectangle(box, outline='green', width=2)

        # Display the image using matplotlib
        plt.figure()
        plt.imshow(img)
        plt.axis('off')
        plt.show()

        display_count += 1