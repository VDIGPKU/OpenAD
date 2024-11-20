import os
import torch
from torch.utils.data import Dataset
import clip
import json
import torchvision.transforms as transforms
import cv2
import open3d as o3d
import numpy as np
import pickle
from .visualize import visualize_pc, visualize_bev, visualize_3d_on_image, visualize_2d_on_image
from .evaluate import get_2d_summary, get_3d_summary


class OpenAD(Dataset):
    """
    OpenAD: Open-World Autonomous Driving Scenes Benchmark for 3D Object Detection
    VDIG @ Peking University

    init args:
    dataroot (str): Path of dataset root.
    training_on (dict): Training on which datasets.
        example: training on nuScenes only.
        training_on = {
            'av2': False,
            'kitti': False,
            'nuscenes': True,
            'once': False,
            'waymo': False,
        }
    frames (int): Including the current frame, __getitem__ returns the number sensor data frames.
        The default value is 1, which means only the current frame is returned.
    """

    def __init__(self, dataroot, training_on=None, frames=1):
        self.dataroot = dataroot
        self.annotations = os.path.join(dataroot, 'annotations')
        self.camera_config = os.path.join(dataroot, 'camera_config')
        self.camera_image = os.path.join(dataroot, 'camera_image')
        self.lidar_point_cloud = os.path.join(dataroot, 'lidar_point_cloud')
        self.training_on = {k.lower(): v for k, v in training_on.items()}
        if 'nusc' in self.training_on.keys() and 'nuscenes' not in self.training_on.keys():
            self.training_on['nuscenes'] = self.training_on['nusc']
        if 'argoverse2' in self.training_on.keys() and 'av2' not in self.training_on.keys():
            self.training_on['av2'] = self.training_on['argoverse2']
        if 'argoverse' in self.training_on.keys() and 'av2' not in self.training_on.keys():
            self.training_on['av2'] = self.training_on['argoverse']
        self.data = list(range(2000))
        self.dataset_list = ['av2', 'kitti', 'nuscenes', 'once', 'waymo']
        self.frames = frames
        self.sample_list = [41, 123, 251, 275, 560, 573, 574, 583, 1749, 1753, 1760, 1765, 1908, 1940, 1970, 1986]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        with open(os.path.join(self.dataroot, 'infos', str(idx) + '.json'), 'r', encoding='utf-8') as file:
            ret_dict = json.load(file)

        img_path = os.path.join(self.dataroot, 'camera_image', str(idx) + '_0.jpg')
        img = cv2.imread(img_path)
        transf = transforms.ToTensor()
        img = transf(img)
        ret_dict['image'] = img
        ret_dict['image_path'] = img_path
        ret_dict['width'] = img.shape[2]
        ret_dict['height'] = img.shape[1]

        pcd = o3d.io.read_point_cloud(os.path.join(self.dataroot, 'lidar_point_cloud', str(idx) + '_0.pcd'))
        points = np.asarray(pcd.points)
        points_tensor = torch.tensor(points, dtype=torch.float32)
        ret_dict['points'] = points_tensor

        lidar_past2now = ret_dict['lidar_past2now']
        lidar_past2now.reverse()

        sweep_list = []
        for sweep_idx_i in range(1, self.frames):
            if sweep_idx_i >= len(lidar_past2now):
                sweep_idx = len(lidar_past2now) - 1
            else:
                sweep_idx = sweep_idx_i
            sweep_item = {}
            img_path = os.path.join(self.dataroot, 'camera_image',
                                    str(idx) + '_' + str(sweep_idx) + '.jpg')
            img = cv2.imread(img_path)
            transf = transforms.ToTensor()
            img = transf(img)
            sweep_item['image'] = img
            sweep_item['image_path'] = img_path

            pcd = o3d.io.read_point_cloud(os.path.join(self.dataroot, 'lidar_point_cloud',
                                                       str(idx) + '_' + str(sweep_idx) + '.pcd'))
            points = np.asarray(pcd.points)
            points_tensor = torch.tensor(points, dtype=torch.float32)
            sweep_item['points'] = points_tensor

            sweep_item['lidar_past2now'] = np.array(lidar_past2now[sweep_idx])

            sweep_list.append(sweep_item)
        ret_dict['sweeps'] = sweep_list

        lidar2cam = ret_dict['camera_external']
        ret_dict['lidar2cam'] = np.array(lidar2cam).reshape(4, 4).T

        return ret_dict

    def visualize_pc(self, idx, save_path='vis_pc.jpg', xylim=(-40, 40, -40, 40)):
        data_dict = self.__getitem__(idx)
        data_dict = self.__get_ann(data_dict, idx)
        visualize_pc(data_dict, save_path=save_path, xylim=xylim)

    def visualize_bev(self, idx, save_path='vis_bev.jpg', xylim=(-40, 40, -20, 60)):
        data_dict = self.__getitem__(idx)
        data_dict = self.__get_ann(data_dict, idx)
        visualize_bev(data_dict, save_path=save_path, xylim=xylim)

    def visualize_2d_on_image(self, idx, save_path='vis_2d_on_image.jpg'):
        data_dict = self.__getitem__(idx)
        data_dict = self.__get_ann(data_dict, idx)
        visualize_2d_on_image(data_dict, save_path=save_path)

    def visualize_3d_on_image(self, idx, save_path='vis_3d_on_image.jpg'):
        data_dict = self.__getitem__(idx)
        data_dict = self.__get_ann(data_dict, idx)
        visualize_3d_on_image(data_dict, save_path=save_path)

    def visualize_pred_bev(self, idx, pred, save_path='vis_pred_bev.jpg', xylim=(-40, 40, -20, 60)):
        data_dict = self.__getitem__(idx)
        for ibox in range(len(pred)):
            ori_box = pred[ibox]
            new_box = [ori_box[7]] + [0] * 7 + ori_box[0:7] + [1]
            pred[ibox] = new_box
        data_dict['annotations'] = pred
        data_dict['seen'] = [None] * len(pred)
        visualize_bev(data_dict, save_path=save_path, xylim=xylim)

    def visualize_pred_2d_on_image(self, idx, pred, save_path='vis_pred_2d_on_image.jpg'):
        data_dict = self.__getitem__(idx)
        for ibox in range(len(pred)):
            ori_box = pred[ibox]
            new_box = [ori_box[4]] + [0] * 3 + ori_box[0:4] + [0] * 7 + [1]
            pred[ibox] = new_box
        data_dict['annotations'] = pred
        data_dict['seen'] = [None] * len(pred)
        visualize_2d_on_image(data_dict, save_path=save_path)

    def visualize_pred_3d_on_image(self, idx, pred, save_path='vis_pred_3d_on_image.jpg'):
        data_dict = self.__getitem__(idx)
        for ibox in range(len(pred)):
            ori_box = pred[ibox]
            new_box = [ori_box[7]] + [0] * 7 + ori_box[0:7] + [1]
            pred[ibox] = new_box
        data_dict['annotations'] = pred
        data_dict['seen'] = [None] * len(pred)
        visualize_3d_on_image(data_dict, save_path=save_path)

    def submit(self, pred_list, save_path='result.pkl'):
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
        assert len(pred_list) == 2000
        for i in range(len(pred_list)):
            if len(pred_list[i]) > 300:
                pred_list[i] = pred_list[i][:300]
                print(
                    f'Number of predicted objects exceeds 300, only the first 300 will be calculated. (data index {i})')
        if type(pred_list[0][0][-1]) is str:
            clip_model, _ = clip.load("ViT-L/14@336px")
            clip_model.cuda()
            text_d = []
            text_feature_d = []
            for d_idx in range(len(pred_list)):
                for d in pred_list[d_idx]:
                    if len(d[-1]) > 75:
                        d[-1] = d[-1][:75]
                    if d[-1] not in text_d:
                        text_d.append(d[-1])
                        text_feature_d.append(clip.tokenize(['a ' + d[-1]]))
            text_feature_d = torch.stack(text_feature_d, dim=0).squeeze(1).cuda()
            text_list = torch.split(text_feature_d, 512, dim=0)
            text_features_list = []
            for i in text_list:
                with torch.no_grad():
                    text_features = clip_model.encode_text(i)
                    text_features = text_features / text_features.norm(dim=1, keepdim=True)
                    text_features_list.append(text_features)
            text_feature_d = torch.cat(text_features_list, dim=0)
            text_feature_d = text_feature_d.cpu().numpy()

            save_list = [pred_list, text_d, text_feature_d, self.training_on]
            with open(save_path, 'wb') as file:
                pickle.dump(save_list, file)
            print("Result file saved at", os.path.abspath(save_path))
        else:
            raise ValueError(f'box[-1] must be str. get {type(pred_list[0][0][-1])}.')

    def evaluate2d(self, pred_list, split='sample'):
        """
        pred_list : list
            A list representing the detected bounding boxes.
            (list)[
                16(num_sample_data) * (list)[
                    N_bboxes * (list)[ (float)x1, y1, x2, y2, (str)c ]
                ]
            ]
        """
        if split == 'sample':
            assert len(pred_list) == 16
            self._evaluate(pred_list, bbox_type='2D')
        elif split == 'full':
            assert len(pred_list) == 2000
            self._evaluate(pred_list, bbox_type='2D')
        else:
            raise ValueError("split must be 'full' or 'sample'")

    def evaluate3d(self, pred_list, split='sample'):
        """
        pred_list : list
            A list representing the detected bounding boxes.
            (list)[
                16(num_sample_data) * (list)[
                    N_bboxes * (list)[ (float)h, w, l, x, y, z, theta, (str)c ]
                ]
            ]
        """
        if split == 'sample':
            assert len(pred_list) == 16
            self._evaluate(pred_list, bbox_type='3D')
        elif split == 'full':
            assert len(pred_list) == 2000
            self._evaluate(pred_list, bbox_type='3D')
        else:
            raise ValueError("split must be 'full' or 'sample'")

    '''
        av2       range(0, 250)
        kitti     range(250, 250 + 309)
        nuscenes  range(250 + 309, 250 + 309 + 134)
        once      range(250 + 309 + 134, 250 + 309 + 134 + 1057)
        waymo     range(250 + 309 + 134 + 1057, 2000)
    '''

    def _cal_dist_volume(self):
        import math
        dist = [[0] * 22, [0] * 22, [0] * 22, [0] * 22, [0] * 22]
        vol = [[0] * 11, [0] * 11, [0] * 11, [0] * 11, [0] * 11]
        vol_split = [0.02, 0.05, 0.1, 0.5, 1, 5, 10, 15, 20, 30]
        start = [0, 250, 250 + 309, 250 + 309 + 134, 250 + 309 + 134 + 1057]
        end = [250, 250 + 309, 250 + 309 + 134, 250 + 309 + 134 + 1057, 2000]
        for i in range(5):
            for isc in range(start[i], end[i]):
                print(isc)
                data_dict = {'width': 1000, 'height': 1000}
                data_dict = self.__get_ann(data_dict, isc)
                anns = data_dict['annotations']
                for parts in anns:
                    h = float(parts[8])
                    w = float(parts[9])
                    l = float(parts[10])
                    x = float(parts[11])
                    z = float(parts[13])
                    d = math.sqrt(x * x + z * z)
                    dd = math.floor(d / 10)
                    vvol = h * w * l
                    vv = 10
                    for vi in range(10):
                        if vvol < vol_split[vi]:
                            vv = vi
                            break
                    dist[i][dd] += 1
                    vol[i][vv] += 1
        return dist, vol

    def __get_ann(self, ret_dict, idx):
        annotations = []
        seen_list = []
        with open(os.path.join(self.dataroot, 'annotations', str(idx) + '.txt'), 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                ann = line.strip().split(' ')[5:]
                seen = line.strip().split(' ')[:5]
                ann[1:] = list(map(float, ann[1:]))
                x1, y1, x2, y2 = ann[4:8]
                x1 = 0 if x1 < 0 else x1
                y1 = 0 if y1 < 0 else y1
                x2 = ret_dict['width'] if x2 > ret_dict['width'] else x2
                y2 = ret_dict['height'] if y2 > ret_dict['height'] else y2
                ann[4:8] = [x1, y1, x2, y2]
                annotations.append(ann)
                training_cnt = 0
                for i, dataset in enumerate(self.dataset_list):
                    if self.training_on[dataset]:
                        training_cnt += int(seen[i])
                if training_cnt > 0:
                    seen_list.append(True)
                else:
                    seen_list.append(False)
        ret_dict['annotations'] = annotations
        ret_dict['seen'] = seen_list
        return ret_dict

    def _evaluate(self, pred_list, bbox_type, split='full'):
        gt_list = []
        sub_list = [0, 0, 0, 0]
        if split == 'full':
            data_list = list(range(2000))
        elif split == 'sample':
            data_list = self.sample_list
        for idx in data_list:
            with open(os.path.join(self.dataroot, 'infos', str(idx) + '.json'), 'r', encoding='utf-8') as file:
                ret_dict = json.load(file)
            ret_dict = self.__get_ann(ret_dict, idx)
            gt = []
            for gtb, seen in zip(ret_dict['annotations'], ret_dict['seen']):
                if seen and self.training_on[ret_dict['dataset']]:
                    subc = 0
                elif seen and not self.training_on[ret_dict['dataset']]:
                    subc = 1
                elif not seen and self.training_on[ret_dict['dataset']]:
                    subc = 2
                else:
                    subc = 3
                sub_list[subc] += 1
                if bbox_type == '2D':
                    gtb = gtb[4:8] + [subc, gtb[0]]
                else:
                    gtb = gtb[8:15] + [subc, gtb[0]]
                gt.append(gtb)
            gt_list.append(gt)

        print('---Evaluate Setting---')
        print(f'type: {bbox_type}, total scene: {len(gt_list)}, '
              f'total objects: {sum(sub_list)}, '
              f'training on {self.training_on}, '
              f'in-domain seen objects: {sub_list[0]}, '
              f'out-domain seen objects: {sub_list[1]}, '
              f'in-domain unseen objects: {sub_list[2]}, '
              f'out-domain unseen objects: {sub_list[3]}.')

        if bbox_type == '2D':
            overall_metrics = get_2d_summary(gt_list, pred_list, sub_list)
        else:
            overall_metrics = get_3d_summary(gt_list, pred_list, sub_list)

        print('---Evaluate Summary---')
        for k, v in overall_metrics.items():
            print(f'{k}: {v:.4f}')

    def _save_gt(self):
        gt_list = []
        for idx in range(2000):
            annotations = []
            with open(os.path.join(self.dataroot, 'infos', str(idx) + '.json'), 'r', encoding='utf-8') as file:
                ret_dict = json.load(file)
            with open(os.path.join(self.dataroot, 'annotations', str(idx) + '.txt'), 'r', encoding='utf-8') as file:
                lines = file.readlines()
            for line in lines:
                ann = line.strip().split(' ')[5:]
                seen = line.strip().split(' ')[:5]
                ann[1:] = list(map(float, ann[1:]))
                x1, y1, x2, y2 = ann[4:8]
                x1 = 0 if x1 < 0 else x1
                y1 = 0 if y1 < 0 else y1
                x2 = ret_dict['width'] if x2 > ret_dict['width'] else x2
                y2 = ret_dict['height'] if y2 > ret_dict['height'] else y2
                ann[4:8] = [x1, y1, x2, y2]
                annotations.append(ann[4:8] + ann[8:15] + seen + [ann[0]])
            gt_list.append(annotations)
        print(gt_list[0][0])

        clip_model, _ = clip.load("ViT-L/14@336px")
        clip_model.cuda()

        text_g = []
        text_feature_g = []
        for g_idx in range(len(gt_list)):
            for g in gt_list[g_idx]:
                if len(g[-1]) > 75:
                    g[-1] = g[-1][:75]
                if g[-1] not in text_g:
                    text_g.append(g[-1])
                    text_feature_g.append(clip.tokenize(['a ' + g[-1]]))

        text_feature_g = torch.stack(text_feature_g, dim=0).squeeze(1).cuda()
        with torch.no_grad():
            text_feature_g = clip_model.encode_text(text_feature_g)
            text_feature_g = text_feature_g / text_feature_g.norm(dim=1, keepdim=True)
        text_feature_g = text_feature_g.cpu().numpy()

        save_list = [gt_list, text_g, text_feature_g]
        with open('gt.pkl', 'wb') as file:
            pickle.dump(save_list, file)
        print("GT saved.")
