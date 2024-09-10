import os
import torch
from torch.utils.data import Dataset
import json
import torchvision.transforms as transforms
import cv2
import open3d as o3d
import numpy as np
from .visualize import visualize_pc, visualize_bev, visualize_3d_on_image, visualize_2d_on_image
from .evaluate import get_2d_summary, get_3d_summary


class OpenAD(Dataset):
    """
    OpenAD: An Open-World Autonomous Driving Scenes Benchmark for 3D Object Detection
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

    def _evaluate(self, pred_list, bbox_type):
        gt_list = []
        in_domain_seen_gt_list = []
        in_domain_unseen_gt_list = []
        out_domain_seen_gt_list = []
        out_domain_unseen_gt_list = []
        in_domain_pred_list = []
        out_domain_pred_list = []
        for idx in range(2000):
            with open(os.path.join(self.dataroot, 'infos', str(idx) + '.json'), 'r', encoding='utf-8') as file:
                ret_dict = json.load(file)
            ret_dict = self.__get_ann(ret_dict, idx)
            seen_gt = []
            unseen_gt = []
            for gtb, seen in zip(ret_dict['annotations'], ret_dict['seen']):
                if bbox_type == '2D':
                    gtb = gtb[4:8] + [gtb[0]]
                else:
                    gtb = gtb[8:15] + [gtb[0]]
                if seen:
                    seen_gt.append(gtb)
                else:
                    unseen_gt.append(gtb)
            if self.training_on[ret_dict['dataset']]:
                in_domain_seen_gt_list.append(seen_gt)
                in_domain_unseen_gt_list.append(unseen_gt)
                in_domain_pred_list.append(pred_list[idx])
                gt_list.append(seen_gt + unseen_gt)
            else:
                out_domain_seen_gt_list.append(seen_gt)
                out_domain_unseen_gt_list.append(unseen_gt)
                out_domain_pred_list.append(pred_list[idx])
                gt_list.append(seen_gt + unseen_gt)

        in_domain_seen_cnt = 0
        in_domain_unseen_cnt = 0
        out_domain_seen_cnt = 0
        out_domain_unseen_cnt = 0
        for s, uns in zip(in_domain_seen_gt_list, in_domain_unseen_gt_list):
            in_domain_seen_cnt += len(s)
            in_domain_unseen_cnt += len(uns)
        for s, uns in zip(out_domain_seen_gt_list, out_domain_unseen_gt_list):
            out_domain_seen_cnt += len(s)
            out_domain_unseen_cnt += len(uns)

        print('---Evaluate Setting---')
        print(f'type: {bbox_type}, total scene: {len(gt_list)}, '
              f'total object: {in_domain_seen_cnt + in_domain_unseen_cnt + out_domain_seen_cnt + out_domain_unseen_cnt}, '
              f'training on {self.training_on}, '
              f'in-domain scene: {len(in_domain_seen_gt_list)}, '
              f'in-domain seen object: {in_domain_seen_cnt}, '
              f'in-domain unseen object: {in_domain_unseen_cnt}, '
              f'out-domain scene: {len(out_domain_seen_gt_list)}, '
              f'out-domain seen object: {out_domain_seen_cnt}, '
              f'out-domain unseen object: {out_domain_unseen_cnt}, '
              )

        if bbox_type == '2D':
            print('OVERALL: ', end='')
            overall_metrics = get_2d_summary(gt_list, pred_list)
            print('IN-DOMAIN SEEN: ', end='')
            in_domain_seen_metrics = get_2d_summary(in_domain_seen_gt_list, in_domain_pred_list)
            print('IN-DOMAIN UNSEEN: ', end='')
            in_domain_unseen_metrics = get_2d_summary(in_domain_unseen_gt_list, in_domain_pred_list)
            print('OUT-DOMAIN SEEN: ', end='')
            out_domain_seen_metrics = get_2d_summary(out_domain_seen_gt_list, out_domain_pred_list)
            print('OUT-DOMAIN UNSEEN: ', end='')
            out_domain_unseen_metrics = get_2d_summary(out_domain_unseen_gt_list, out_domain_pred_list)
        else:
            print('OVERALL: ', end='')
            overall_metrics = get_3d_summary(gt_list, pred_list)
            print('IN-DOMAIN SEEN: ', end='')
            in_domain_seen_metrics = get_3d_summary(in_domain_seen_gt_list, in_domain_pred_list)
            print('IN-DOMAIN UNSEEN: ', end='')
            in_domain_unseen_metrics = get_3d_summary(in_domain_unseen_gt_list, in_domain_pred_list)
            print('OUT-DOMAIN SEEN: ', end='')
            out_domain_seen_metrics = get_3d_summary(out_domain_seen_gt_list, out_domain_pred_list)
            print('OUT-DOMAIN UNSEEN: ', end='')
            out_domain_unseen_metrics = get_3d_summary(out_domain_unseen_gt_list, out_domain_pred_list)

        print('---Evaluate Summary---')
        print(f'OpenAD\t\t\tAP ↑ \tAR ↑ \tATE ↓\tASE ↓')
        for title, m in zip(['in-domain seen\t', 'in-domain unseen', 'out-domain seen\t',
                             'out-domain unseen', 'overall\t\t'],
                            [in_domain_seen_metrics, in_domain_unseen_metrics, out_domain_seen_metrics,
                             out_domain_unseen_metrics, overall_metrics]):
            print(title, end='')
            for item in ['AP', 'AR', 'ATE', 'ASE']:
                print(f'\t{m[item]:.4f}', end='')
            print()

    def evaluate2d(self, pred_list):
        """
        pred_list : list
            A list representing the detected bounding boxes.
            (list)[
                2000 * (list)[
                    N_bboxes * (list)[ (float)x1, y1, x2, y2, (str)c ]
                ]
            ]
        """
        assert len(pred_list) == 2000
        self._evaluate(pred_list, bbox_type='2D')

    def evaluate3d(self, pred_list):
        """
        pred_list : list
            A list representing the detected bounding boxes.
            (list)[
                2000 * (list)[
                    N_bboxes * (list)[ (float)h, w, l, x, y, z, theta, (str)c ]
                ]
            ]
        """
        assert len(pred_list) == 2000
        self._evaluate(pred_list, bbox_type='3D')
