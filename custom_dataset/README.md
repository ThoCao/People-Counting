### Custom Dataset for Training Models with Combined COCO and CrowdHuman Data
### Goals
This codebase downloads people object datasets from multiple sources and converts them to YOLO format for custom training, using [COCO](https://cocodataset.org/#home) and [CrowdHuman](https://www.crowdhuman.org/) for fine-tuning a YOLO model. 
### Design
The architecture of this codebase currently supports COCO and CrowdHuman datasets. However, its design allows for managing multiple dataset sources. Below is a detailed overview of the classes designed in this module.
* **DatasetManager**:  This class performs several tasks, including downloading, extracting, and creating labels from each configuration file.
* **CustomDataset**: This class, tailored for people detection, inherits from PyTorch's Dataset to facilitate data manipulation and seamless integration with PyTorch's DataLoader. 
* **Utils**: Offers functions for visualizing and verifying the dataset. 
* **Config**: Defines the dataset configuration, including the number of samples and dataset source information. It uses [Hydra](https://hydra.cc/) framework.

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
│   │   ├── 0011.txt
│   │   └── ...
│   └── val2017
│       ├── 0011.jpg
│       └── ...
├── crowdhuman
│   ├── annotations
│   ├── Image
│   │   ├── 0012.jpg
│   │   └── ...
│   └── labels
│       ├── 0012.txt
│       └── ...
├── data_manager.py
├── custom_datasets.py
├── my_utils.py
├── main.py
└── demo_data.ipynb
```
### Demo
* Google Colab Execute: [colab demo]
* Local Execuation
To run an experiment, configure the **config.yaml** for the Hydra settings:
```
python main.py
```