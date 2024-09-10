# OpenAD: An Open-World Autonomous Driving Scenes Benchmark for 3D Object Detection

<img src="https://github.com/VDIGPKU/OpenAD/blob/main/assets/openad.jpg" width="1000"/>

OpenAD is the first open-world 3D object detection benchmark for autonomous driving. 
We selected 2,000 scenes from 5 public datasets and annotated 6,597 3D corner cases for each scene. 
You can use this toolkit to organize data, load data, and evaluate your model with a few simple commands.

## Update

* 2024/9/10 - We have released OpenAD, and it is currently in beta. 
We welcome your feedback and suggestions for using this benchmark.

## Introduction

|  Dataset   | Scenes | Original Obj | Added Obj | Total Obj | Seen  | Unseen |
|:----------:|:------:|:------------:|:---------:|:---------:|:-----:|:------:|
| Argoverse2 |  250   |     5422     |    552    |   5974    | 18220 |  1541  |
|   KITTI    |  309   |     1902     |    363    |   2265    | 9596  | 10165  | 
|  nuScenes  |  134   |     2019     |    581    |   2600    | 14739 |  5022  | 
|    ONCE    |  1057  |     157      |   4298    |   4455    | 10706 |  9055  | 
|   Waymo    |  250   |     3664     |    803    |   4467    | 12557 |  7204  | 
|   Total    |  2000  |    13164     |   6597    |   19761   |       |        | 

## Data Preparation

It is recommended to create a new Python virtual environment to prepare the complete openad data.

1. Due to the use of terms of each dataset, 
we cannot provide downloads of the full data. 
You need to download the dataset [nuScenes](https://www.nuscenes.org/nuscenes), 
[Argoverse2](https://www.argoverse.org/av2.html), 
[Kitti](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d), 
[ONCE](https://once-for-auto-driving.github.io/) and 
[Waymo](https://waymo.com/open/), 
download the newly added annotation and index file (openad.zip
[Google Drive](https://drive.google.com/file/d/1DVguskUEqgynt6ZET54vmV4sGvhWtw40/view?usp=drive_link)
or
[PKU Drive](https://disk.pku.edu.cn/link/AAB6C49977A1F247D5BC8DA923EB030D54)
).

2. Install [PyTorch](https://pytorch.org/get-started/locally/) and [CLIP](https://github.com/openai/CLIP).

3. Install requirements and install this toolkit.

```bash
git clone git@github.com:VDIGPKU/OpenAD.git
cd OpenAD
pip install -r requirements_create.txt
python setup.py install
```

4. Create the complete openad data

```bash
python -c "
from openad.create import create_openad; 
create_openad(av2_root='/path/to/argoverse2', 
    kitti_root='/path/to/kitti_for_detection', 
    nuscenes_root='/path/to/nuscenes',
    once_root='/path/to/once',
    waymo_root='/path/to/waymo',
    openad_zip='/path/to/openad.zip',
    openad_root='/path/to/save/openad'
)"
```

If you encounter any difficulties in creating data, 
please raise an issue or send an email to xiazhongyu@pku.edu.cn

## Getting Started

### Install OpenAD Toolkit

1. Install [PyTorch](https://pytorch.org/get-started/locally/) and [CLIP](https://github.com/openai/CLIP).

2. Install requirements and install this toolkit.

```bash
git clone git@github.com:VDIGPKU/OpenAD.git
cd OpenAD
pip install -r requirements.txt
python setup.py install
```

### Loading Data

You can directly use the already defined torch.dataset subclass to load the data.

An example is shown below, 
where the model to be tested is trained only on the nuScenes training set,
and the model requires the current frame and the past 4 frames (out of 5 frames) as input.

```python
from openad import OpenAD

dd = OpenAD(
    dataroot='/path/to/openad',
    training_on={
        'av2': False,
        'kitti': False,
        'nuscenes': True,
        'once': False,
        'waymo': False,
    },
    frames=5
)

print(dd[1970].keys())
```
```
dict_keys(['width', 'height', 'rowMajor', 'camera_internal', 'camera_external', 'dataset', 'timestamp_ms', 
'sensors_path', 'lidar_past2now', 'image', 'image_path', 'points', 'sweeps', 'lidar2cam'])
```

### Evaluation

**Please make sure ‘training_on’ is correct and make sure the model under test is not trained on the validation set of all five datasets.**

You can directly evaluate your 2D or 3D detection results by:

```python
dd.evaluate2d(pred_list)
"""
pred_list : list
    A list representing the detected bounding boxes.
    (list)[
        2000 * (list)[
            N_bboxes * (list)[ (float)x1, y1, x2, y2, (str)c ]
        ]
    ]
"""

dd.evaluate3d(pred_list)
"""
pred_list : list
    A list representing the detected bounding boxes.
    (list)[
        2000 * (list)[
            N_bboxes * (list)[ (float)h, w, l, x, y, z, theta, (str)c ]
        ]
    ]
"""
```

For the 2D bounding box, we followed the calculation method of pycocotools, 
using the clip threshold of {0.5, 0.7, 0.9} and the IoU threshold of {0.5, 0.55, 0.6, ..., 0.85, 0.9} 
to determine whether it is TP or not and calculate the AP and AR.

For the 3D bounding box, we followed the calculation method of nuScenes benchmark, 
using the clip threshold of {0.5, 0.7, 0.9} and the center distance threshold of {0.5m, 1.0m, 2.0m, 4.0m} 
to determine whether it is TP or not and calculate the AP and AR.

When calculating the above TP metrics, 
inspired by nuScenes benchmark, this program will also calculate the regression metrics ATE and ASE.
Average Translation Error (ATE) is euclidean center distance in pixels (2D) or meters (3D).
Average Scale Error (ASE) is calculated as 1 - IOU after aligning centers and orientation.

In addition, the evaluation program will divide the evaluation data into four parts, 
according to whether it belongs to the training data set 
and whether the object category has been seen during training. 
the evaluation program will then calculate the above metrics respectively.
This can evaluate the model's open-scene domain-adaptation ability and open-vocabulary ability respectively.

### Visualization

We provide visualization tools as shown below.

```python
dd.visualize_2d_on_image(1970)  # show 2D GT bbox on image 
```
<img src="https://github.com/VDIGPKU/OpenAD/blob/main/assets/vis_2d_on_image.jpg" width="400"/>

```python
dd.visualize_3d_on_image(1970)  # show 3D GT bbox on image 
```
<img src="https://github.com/VDIGPKU/OpenAD/blob/main/assets/vis_3d_on_image.jpg" width="400"/>

```python
dd.visualize_bev(1970)  # show 3D GT bbox and point clouds under BEV
```
<img src="https://github.com/VDIGPKU/OpenAD/blob/main/assets/vis_bev.jpg" width="400"/>

```python
dd.visualize_pc(1970)  # show aligned multi-frame point clouds under BEV
```
<img src="https://github.com/VDIGPKU/OpenAD/blob/main/assets/vis_pc.jpg" width="400"/>

## Licenses

Unless specifically labeled otherwise, these code and OpenAD dataset files are provided to You 
under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License 
(“[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)”), 
with the additional terms included herein.
When You download or use the Datasets from the Website or elsewhere, 
you are agreeing to comply with the terms of CC BY-NC-SA 4.0, 
and also agreeing to the Dataset Terms. 
Where these Dataset Terms conflict with the terms of CC BY-NC-SA 4.0, 
these Dataset Terms shall prevail. 
We reiterate once again that this dataset is used only for non-commercial purposes such as academic research, 
teaching, or scientific publications. 
We prohibit you from using the dataset or any derivative works for commercial purposes, 
such as selling data or using it for commercial gain.
