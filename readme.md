# OpenAD: An Open-World Autonomous Driving Scenes Benchmark for 3D Object Detection

<img src="https://github.com/VDIGPKU/OpenAD/blob/main/assets/openad.jpg" width="1000"/>

## Update

* 2024/xx/xx - The online evaluation for OpenAD has been opened. Test your model on [EvalAI - OpenAD 2D](https://eval.ai/web/challenges/challenge-page/2416/overview) and [EvalAI - OpenAD 3D](https://eval.ai/web/challenges/challenge-page/2414/overview)!

* 2024/xx/xx - We have released our paper on [arXiv](TODO).

* 2024/9/10 - We have released OpenAD, which is currently in beta. 
We welcome your feedback and suggestions for using this benchmark.

## Introduction

<img src="https://github.com/VDIGPKU/OpenAD/blob/main/assets/properties.png" width="1000"/>

OpenAD is the first open-world 3D object detection benchmark for autonomous driving. 
We meticulously selected 2,000 scenes from 5 public datasets and annotated 6,597 3D corner cases for these scenes. 
Together with the original annotations of these scenes, there are 19,761 objects belonging to 206 different categories.

You can utilize OpenAD to evaluate your model's open-world capabilities, 
encompassing scene generalization, cross-vehicle-type adaptability, open-vocabulary proficiency, and corner case detection aptitude.

You can use this toolkit to organize data, load data, and evaluate your model with simple commands.

## Data Preparation

Creating a new Python virtual environment is recommended to prepare the complete OpenAD data.

1. Due to the use of terms of each dataset, 
we cannot provide downloads of the full data. 
You need to download the dataset [nuScenes](https://www.nuscenes.org/nuscenes), 
[Argoverse2](https://www.argoverse.org/av2.html), 
[Kitti](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d), 
[ONCE](https://once-for-auto-driving.github.io/) and 
[Waymo](https://waymo.com/open/), 
download the newly added annotation and index file (openad.zip
[Google Drive](https://drive.google.com/file/d/1u1k1-FSjK_ezttx5AS8FuabkRybc1meA/view?usp=drive_link)
or
[PKU Drive](https://disk.pku.edu.cn/link/AA49F9C955E80240988567387249E82E09).

2. Install [PyTorch](https://pytorch.org/get-started/locally/) and [CLIP](https://github.com/openai/CLIP).

3. Install requirements and install this toolkit.

```bash
git clone git@github.com:VDIGPKU/OpenAD.git
cd OpenAD
pip install -r requirements_create.txt
python setup.py install
```

4. Create the complete OpenAD data

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
please raise an issue or email xiazhongyu@pku.edu.cn

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

You can directly use the already-defined torch.dataset subclass to load the data.

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
print(len(dd[1970]['sweeps']))
print(dd[1970]['sweeps'][0].keys())
```
```
dict_keys(['width', 'height', 'rowMajor', 'camera_internal', 'camera_external',
'dataset', 'timestamp_ms', 'sensors_path', 'lidar_past2now', 'image',
'image_path', 'points', 'sweeps', 'lidar2cam'])
4
dict_keys(['image', 'image_path', 'points', 'lidar_past2now'])
```

### Evaluation

**Please make sure ‘training_on’ is correct and make sure the model under test is not trained on the validation set of all five datasets.**

You can utilize the following tools to package your prediction results into a specific format and submit them for [online evaluation](TODO):

```python
dd.submit(pred_list, save_path='result.pkl')

"""
2D Track: pred_list (list)
    A list representing the detected bounding boxes.
    (list)[
        2000 * (list)[
            N_bboxes * (list)[ (float)x1, y1, x2, y2, (str)c ]
        ]
    ]

3D Track: pred_list (list)
    A list representing the detected bounding boxes.
    (list)[
        2000 * (list)[
            N_bboxes * (list)[ (float)h, w, l, x, y, z, theta, (str)c ]
        ]
    ]
"""
```

You can directly evaluate your 2D or 3D detection results on sample data by:

```python
dd.evaluate2d(pred_list)
"""
pred_list : list
    A list representing the detected bounding boxes.
    (list)[
        16(num_samples) * (list)[
            N_bboxes * (list)[ (float)x1, y1, x2, y2, (str)c ]
        ]
    ]
"""

dd.evaluate3d(pred_list)
"""
pred_list : list
    A list representing the detected bounding boxes.
    (list)[
        16(num_samples) * (list)[
            N_bboxes * (list)[ (float)h, w, l, x, y, z, theta, (str)c ]
        ]
    ]
"""
```

#### Average Precision (AP) and Average Recall (AR)

The calculation of AP and AR depends on True Positive (TP).
In OpenAD, the threshold of TP incorporates both positional and semantic scores.
An object prediction is considered a TP only if it simultaneously meets both the positional and semantic thresholds.
For 2D object detection, in line with COCO, Intersection over Union (IoU) is used as the positional score. 
We use the cosine similarity of features from the CLIP model as the semantic score.
When calculating AP, IoU thresholds ranging from 0.5 to 0.95 with a step size of 0.05 are used, along with semantic similarity thresholds of 0.5, 0.7, and 0.9.

For 3D object detection, the center distance is adopted as the positional score following nuScenes, and we use the same semantic score as the 2D detection task.
Similar to nuScenes, we adopt a multi-threshold averaging method for AP calculation.
Specifically, we compute AP across 12 thresholds, combining positional thresholds of 0.5m, 1m, 2m, and 4m with semantic similarity thresholds of 0.5, 0.7, and 0.9, and then average these AP values.

The same principle applies to calculating Average Recall (AR) for 2D and 3D object detection tasks.
Both AP and AR are calculated only for the top 300 predictions.

#### Average Translation Error (ATE) and Average Scale Error (ASE) 

Following nuScenes, we also evaluate the prediction quality of TP objects using regression metrics.
The Average Translation Error (ATE) refers to the Euclidean center distance, measured in pixels for 2D or meters for 3D.
The Average Scale Error (ASE) is calculated as `1 - IoU` after aligning the centers and orientations of the predicted and ground truth objects.

#### In/Out Domain & Seen/Unseen AR

To evaluate the model's domain generalization ability and open-vocabulary capability separately, we calculate the AR based on whether the scene is within the training domain and whether the object semantics have been seen during training.
The positional thresholds for this metric are defined as above, whereas the semantic similarity thresholds are fixed at 0.9.

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

You can also visualize the predictions by:

```python
dd.visualize_pred_2d_on_image(idx, your_prediction)
"""
your_prediction:
(list)[
    N_bboxes * (list)[ (float)x1, y1, x2, y2, (str)c ]
]
"""

dd.visualize_pred_3d_on_image(idx, your_prediction)
dd.visualize_pred_bev(idx, your_prediction)
"""
your_prediction:
(list)[
    N_bboxes * (list)[ (float)h, w, l, x, y, z, theta, (str)c ]
]
"""
```

## Licenses

Unless specifically labeled otherwise, this toolkit code and OpenAD dataset files are provided to You 
under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License 
(“[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)”), 
with the additional terms included herein.
When You download or use the Datasets from the Website or elsewhere, 
you are agreeing to comply with the terms of CC BY-NC-SA 4.0, 
and also agreeing to the Dataset Terms. 
This toolkit code and OpenAD dataset files are used only for non-commercial purposes such as academic research, 
teaching, or scientific publications. 
For business cooperation, please contact wyt@pku.edu.cn.
