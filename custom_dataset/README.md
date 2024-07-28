### Custom Dataset for Training Models with Combined COCO and CrowdHuman Data
#### Project_directory
```
├── config
│   ├── config.yaml
│   └── dataset
│       ├── coco.yaml
│       └── crowdhuman.yaml
├── coco
│   ├── annotations
│   ├── labels
│   │   ├── 001.txt
│   │   └── ...
│   └── val2017
│       ├── 001.jpg
│       └── ...
├── crowdhuman
│   ├── annotations
│   ├── Image
│   │   ├── 001.jpg
│   │   └── ...
│   └── labels
│       ├── 001.txt
│       └── ...
├── data_manager.py
├── custom_datasets.py
├── my_utils.py
├── main.py
└── demo_data.ipynb
```
