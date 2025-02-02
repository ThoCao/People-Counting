# Unique-People-Counting
**Author(s):**
* Tho Cao

**Status:** Implementing
![People Counting Output](media/frame_1.png)
### Goals
This codebase provides the experiment configs, analytics scripts to test and evaluate models for counting unique-people, as well export result to another format (e.g. debug images, csv files)
### Report
[Doc Report](https://docs.google.com/document/d/1EiTHYLZ3Glm-0KqSXysIZBPLFRTlHXsQCsbUflOEVsQ/edit?usp=sharing)
### Installation
`conda` virtual environment is recommended. 
```
conda create -n people_counting python=3.9
conda activate people_counting
pip install -r requirements.txt
pip install -e .
```
**Problems with the OpenCV library for displaying images** [option]
1. Install dependencies:
```
sudo apt-get update
sudo apt-get install libgtk2.0-dev pkg-config
sudo apt-get install libgl1-mesa-glx
```
2. Uninstall the current Opencv
```
pip uninstall opencv-python
```
3. Reinstall OpenCV
```
pip install opencv-python
```
### Experiment Configuration
We use [Hydra](https://hydra.cc/) for managing experiment configs in hierachical and reuable fashion. 
#### Basic Concepts
This section includes specific details on how we use Hydra. For example, we define a configuration for evaluating the performance of multiple models by adding the model names that we want to evaluate and some basic configuration for input and output paths. Here is an example for **'eval_config.yaml'**
```yaml
defaults:
  - _self_
base_path: "./test_results"
ground_truth_csv: "ground_truth.csv"
output_csv: "output_metrics.csv"
model_names:
  - "yolov5n"
  - "yolov5s"
  - "yolov10x"
  - "yolov10n"
```
#### Demo
* Gogole Colab Execution: Access the [colab demo](https://colab.research.google.com/drive/1s0N4sLrBQcTm7MYXuGjwRC9QlGCbpWz0?usp=sharing)
* Local Execuation

To run an experiment on a video, configure the **demo_config.yaml** for the Hydra settings:
```
python demo_net.py
```
![Net Demo Output](media/net_demo_output.png)
* **Debug images:** A bounding box is drawn around each detected person in the video, and the count of unique individuals is updated at each frame.
* **Debug video:** A video with drawn bounding boxes.
* **CSV file:** Contains results storing the number of unique-people is updated at each frame.

To evaluate model performance, configure the **eval_config.yaml** for the Hydra settings:
```
python eval_net.py
```
![Model Eval Output](media/model_eval_output.png)
#### Custom Training 
* **Preparing custom dataset**: Please follow the instructions in the [custom_dataset](https://github.com/ThoCao/People-Counting/tree/main/custom_dataset) folder to download the people dataset.
* **Train custom data**: It is recommended to use the [original tutorial](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/).

For the yolov5 model:
```
python train.py --img 640 --epochs 3 --data coco128.yaml --weights yolov5s.pt
```
For the yolov10 model:
```
yolo detect train data=coco.yaml model=yolov10n/s/m/b/l/x.yaml epochs=500 batch=256 imgsz=640 device=0,1,2,3,4,5,6,7
```
### Future Work

### References
* [YOLOv10: Real-Time End-to-End Object Detection (paper)](https://arxiv.org/abs/2405.14458)
* [YOLOv10: Source Code](https://github.com/THU-MIG/yolov10?tab=readme-ov-file)
* [YOLOv5: Source Code](https://github.com/ultralytics/yolov5)