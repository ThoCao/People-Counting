import os
from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data import Dataset
import torch
from PIL import Image

class CocoDataset(Dataset):
    def __init__(self, img_ids: List[int], dataset: Any, image_dir: str, transform: Optional[Any] = None) -> None:
        self.img_ids = img_ids
        self.dataset = dataset
        self.image_dir = image_dir
        self.transform = transform
        self.person_cat_id = self.dataset.getCatIds(catNms=['person'])[0]

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
        img_id = self.img_ids[idx]
        ann_ids = self.dataset.getAnnIds(imgIds=img_id, catIds=[self.person_cat_id], iscrowd=None)
        anns = self.dataset.loadAnns(ann_ids)

        img_info = self.dataset.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')

        boxes = [ann['bbox'] for ann in anns if 'bbox' in ann and len(ann['bbox']) == 4]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        target = {'boxes': boxes, 'labels': torch.ones((len(boxes),), dtype=torch.int64)}

        if self.transform:
            img = self.transform(img)

        return img, target

class CrowdHumanDataset(Dataset):
    def __init__(self, img_ids: List[str], dataset: Dict[str, Any], image_dir: str, transform: Optional[Any] = None) -> None:
        self.img_ids = img_ids
        self.dataset = dataset
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
        img_id = self.img_ids[idx]
        anns = [ann for ann in self.dataset['annotations'] if ann['image_id'] == img_id]

        img_path = os.path.join(self.image_dir, f'{img_id}.jpg')
        img = Image.open(img_path).convert('RGB')

        boxes = [ann['bbox'] for ann in anns]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        target = {'boxes': boxes, 'labels': torch.ones((len(boxes),), dtype=torch.int64)}

        if self.transform:
            img = self.transform(img)

        return img, target