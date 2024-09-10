import os
import numpy as np
import open3d as o3d
from PIL import Image
import av2.structures.cuboid as av2cub


def save_av2(av2_root, img_path_list, pc_path_list, openad_root, idx):
    for frame_idx, item in enumerate(reversed(img_path_list)):
        img = Image.open(os.path.join(av2_root, item))
        img.save(os.path.join(openad_root, 'camera_image', f'{idx}_{frame_idx}.jpg'))
    for frame_idx, item in enumerate(reversed(pc_path_list)):
        points = av2cub.read_feather(os.path.join(av2_root, item))
        points = np.array(points, dtype=np.float32)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        save_path = os.path.join(openad_root, 'lidar_point_cloud', f'{idx}_{frame_idx}.pcd')
        o3d.io.write_point_cloud(save_path, pcd)
