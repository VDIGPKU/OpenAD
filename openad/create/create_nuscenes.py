from PIL import Image
import os
from nuscenes.utils.data_classes import LidarPointCloud
import open3d as o3d


def save_nuscenes(nuscenes_root, img_path_list, pc_path_list, openad_root, idx):
    for frame_idx, item in enumerate(reversed(img_path_list)):
        img = Image.open(os.path.join(nuscenes_root, item))
        img.save(os.path.join(openad_root, 'camera_image', f'{idx}_{frame_idx}.jpg'))
    for frame_idx, item in enumerate(reversed(pc_path_list)):
        points = LidarPointCloud.from_file(os.path.join(nuscenes_root, item))
        points = points.points.T
        points = points.reshape((-1, 4))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        save_path = os.path.join(openad_root, 'lidar_point_cloud', f'{idx}_{frame_idx}.pcd')
        o3d.io.write_point_cloud(save_path, pcd)