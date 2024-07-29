import os
import zipfile
import json
import gdown
import random
from typing import List, Dict, Any, Optional
from pycocotools.coco import COCO
from PIL import Image
from custom_datasets import CocoDataset, CrowdHumanDataset
from torch.utils.data import Dataset

class DatasetManager:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        dataset_cfg = cfg['dataset']  # Access the nested dataset configuration
        self.data_dir = dataset_cfg['data_dir']
        self.annotations_dir = dataset_cfg['annotations_dir']
        self.image_dir = dataset_cfg['image_dir']
        self.annotations_file = dataset_cfg['annotations_file']
        self.dataset_type = dataset_cfg['dataset_type']
        self.label_dir = os.path.join(self.data_dir, 'labels')  # Initialize label_dir

        self._create_directories()
        self._download_and_extract_files(dataset_cfg['files_to_download'])
    
        if self.dataset_type == 'COCO':
            if os.path.exists(self.annotations_file):
                self.dataset = COCO(self.annotations_file)
            else:
                raise FileNotFoundError(f"COCO annotations file not found: {self.annotations_file}")
        elif self.dataset_type == 'CrowdHuman':
            self.dataset = self._load_crowdhuman_annotations()

    def _create_directories(self)-> None:
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)

    def _download_and_extract_files(self, files_to_download: Dict[str, str]) -> None:
        for file_name, url in files_to_download.items():
            file_path = os.path.join(self.data_dir, file_name)
            if not os.path.exists(file_path):
                print(f'Downloading {file_name}...')
                gdown.download(url, file_path, quiet=False)

            if file_name.endswith('.zip') and os.path.exists(file_path):
                print(f'Extracting {file_name}...')
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    self._extract_to_target(zip_ref, file_name)

    def _extract_to_target(self, zip_ref, file_name):
        if self.dataset_type == 'COCO':
            target_path = self.annotations_dir if 'annotations' in file_name else self.image_dir
        elif self.dataset_type == 'CrowdHuman':
            if 'annotations' in file_name:
                target_path = self.annotations_dir
            elif 'Images' in file_name:
                target_path = self.image_dir
            else:
                target_path = self.label_dir
        else:
            target_path = self.data_dir

        # Extract files directly to the target path without creating extra subfolders
        for member in zip_ref.namelist():
            filename = os.path.basename(member)
            if not filename:
                continue
            source = zip_ref.open(member)
            # Remove leading directory components from the member name
            relative_path = os.path.relpath(member, start=os.path.commonpath(zip_ref.namelist()))
            if 'annotations' in file_name:
                target = os.path.join(self.annotations_dir, os.path.basename(relative_path))
            else:
                target = os.path.join(self.image_dir, os.path.basename(relative_path))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "wb") as f:
                with source as src:
                    f.write(src.read()) 

    def _load_crowdhuman_annotations(self) -> Dict[str, Any]:
        with open(self.annotations_file, 'r') as f:
            lines = f.readlines()
            annotations = [json.loads(line) for line in lines if line.strip()]

        dataset = {'images': [], 'annotations': [], 'categories': []}
        for ann in annotations:
            dataset['images'].append({'id': ann['ID'], 'file_name': ann['ID'] + '.jpg'})
            for gtbox in ann['gtboxes']:
                if gtbox['tag'] == 'person' and gtbox['extra'].get('ignore', 0) == 0:
                    dataset['annotations'].append({
                        'image_id': ann['ID'],
                        'bbox': gtbox['fbox'],
                        'category_id': 1
                    })
        dataset['categories'].append({'id': 1, 'name': 'person'})
        return dataset

    def get_img_ids(self, category_name: Optional[str] = None, sample_size: int = 50) -> List[int]:
        if self.dataset_type == 'COCO':
            cat_ids = self.dataset.getCatIds(catNms=[category_name]) if category_name else []
            img_ids = self.dataset.getImgIds(catIds=cat_ids)
        elif self.dataset_type == 'CrowdHuman':
            img_ids = [img['id'] for img in self.dataset['images']]
        print(f"Total images available: {len(img_ids)}")
        return random.sample(img_ids, min(sample_size, len(img_ids)))

    def create_dataset(self, img_ids: List[int], transform: Optional[Any] = None) -> Dataset:
        if self.dataset_type == 'COCO':
            return CocoDataset(img_ids, self.dataset, self.image_dir, transform)
        elif self.dataset_type == 'CrowdHuman':
            return CrowdHumanDataset(img_ids, self.dataset, self.image_dir, transform)

    def save_labels(self, img_ids: List[int]) -> None:
        if self.dataset_type == 'COCO':
            dataset = CocoDataset(img_ids, self.dataset, self.image_dir)
            for img_id in img_ids:
                img_info = self.dataset.loadImgs(img_id)[0]
                img_path = os.path.join(self.image_dir, img_info['file_name'])
                img_width, img_height = Image.open(img_path).size

                ann_ids = self.dataset.getAnnIds(imgIds=img_id, catIds=[self.dataset.getCatIds(catNms=['person'])[0]], iscrowd=None)
                anns = self.dataset.loadAnns(ann_ids)
                boxes = [(ann['bbox'], ann['category_id']) for ann in anns if 'bbox' in ann and len(ann['bbox']) == 4]

                label_file_path = os.path.join(self.label_dir, f"{img_info['file_name'].split('.')[0]}.txt")
                with open(label_file_path, 'w') as f:
                    for box, category_id in boxes:
                        x_center = (box[0] + box[2] / 2) / img_width
                        y_center = (box[1] + box[3] / 2) / img_height
                        width = box[2] / img_width
                        height = box[3] / img_height
                        f.write(f"{category_id} {x_center} {y_center} {width} {height}\n")

        elif self.dataset_type == 'CrowdHuman':
            for img_id in img_ids:
                img_info = next((img for img in self.dataset['images'] if img['id'] == img_id), None)
                img_path = os.path.join(self.image_dir, img_info['file_name'])
                img_width, img_height = Image.open(img_path).size

                anns = [ann for ann in self.dataset['annotations'] if ann['image_id'] == img_id]
                boxes = [(ann['bbox'], ann['category_id']) for ann in anns]

                label_file_path = os.path.join(self.label_dir, f"{img_info['file_name'].split('.')[0]}.txt")
                with open(label_file_path, 'w') as f:
                    for box, category_id in boxes:
                        x_center = (box[0] + box[2] / 2) / img_width
                        y_center = (box[1] + box[3] / 2) / img_height
                        width = box[2] / img_width
                        height = box[3] / img_height
                        f.write(f"{category_id} {x_center} {y_center} {width} {height}\n")