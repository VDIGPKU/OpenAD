import os
import numpy as np
import open3d as o3d
from PIL import Image


def save_kitti(kitti_root, img_path_list, pc_path_list, openad_root, idx):
    for frame_idx, item in enumerate(reversed(img_path_list)):
        img = Image.open(os.path.join(kitti_root, item))
        img.save(os.path.join(openad_root, 'camera_image', f'{idx}_{frame_idx}.jpg'))
    for frame_idx, item in enumerate(reversed(pc_path_list)):
        points = np.fromfile(os.path.join(kitti_root, item), dtype=np.float32).reshape((-1, 4))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        save_path = os.path.join(openad_root, 'lidar_point_cloud', f'{idx}_{frame_idx}.pcd')
        o3d.io.write_point_cloud(save_path, pcd)
