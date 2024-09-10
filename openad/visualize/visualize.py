import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from .utils import Object3d, compute_box_3d, draw_2d_box, draw_projected_box3d


def visualize_pc(data_dict, save_path='vis_pc.jpg', xylim=(-40, 40, -40, 40)):
    """Plot Bird's Eye View of multi-frame point cloud"""
    fig, ax = plt.subplots(figsize=(20, 20))
    point_cloud = data_dict['points']
    lidar2cam = data_dict['lidar2cam']
    points_hom = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
    points_transformed = points_hom.dot(lidar2cam.T)
    point_cloud = points_transformed[:, :3]
    ax.scatter(point_cloud[:, 0], point_cloud[:, 2], s=0.1, c='gray', alpha=0.5)
    for sweep_item in data_dict['sweeps']:
        point_cloud = sweep_item['points']
        points_hom = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
        lidar_past2now = sweep_item['lidar_past2now']
        points_transformed = points_hom.dot(lidar_past2now.T).dot(lidar2cam.T)
        point_cloud = points_transformed[:, :3]
        ax.scatter(point_cloud[:, 0], point_cloud[:, 2], s=0.1, c='gray', alpha=0.5)
    ax.set_xlim(xylim[0], xylim[1])
    ax.set_ylim(xylim[2], xylim[3])
    plt.savefig(save_path)
    print(f"Point Cloud BEV saved at {save_path}")


def visualize_bev(data_dict, save_path='vis_bev.jpg', xylim=(-40, 40, -20, 60)):
    """Plot Bird's Eye View of point cloud and 3D boxes"""
    fig, ax = plt.subplots(figsize=(20, 20))
    point_cloud = data_dict['points']
    lidar2cam = data_dict['lidar2cam']
    points_hom = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
    points_transformed = points_hom.dot(lidar2cam.T)
    point_cloud = points_transformed[:, :3]
    ax.scatter(point_cloud[:, 0], point_cloud[:, 2], s=0.1, c='gray', alpha=0.5)
    anns = data_dict['annotations']
    boxes = []
    for parts in anns:
        box = {
            'type': parts[0],
            'truncated': float(parts[1]),
            'occluded': int(parts[2]),
            'alpha': float(parts[3]),
            'bbox': [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])],
            'dimensions': [float(parts[8]), float(parts[9]), float(parts[10])],
            'location': [float(parts[11]), float(parts[12]), float(parts[13])],
            'rotation_y': float(parts[14])
        }
        boxes.append(box)
    for seen, box in zip(data_dict['seen'], boxes):
        h, w, l = box['dimensions']
        x, y, z = box['location']
        rot = box['rotation_y']
        # ax.scatter(x, z, c='red', marker='x')
        # Compute the 4 corners of the box
        corners = np.array([
            [l / 2, 0, w / 2],
            [-l / 2, 0, w / 2],
            [-l / 2, 0, -w / 2],
            [l / 2, 0, -w / 2]
        ])
        rot_matrix = np.array([
            [np.cos(rot), 0, np.sin(rot)],
            [0, 1, 0],
            [-np.sin(rot), 0, np.cos(rot)]
        ])
        corners = np.dot(corners, rot_matrix.T) + np.array([x, y, z])
        if seen:
            rect = plt.Polygon(corners[:, [0, 2]], fill=None, edgecolor='g')
        else:
            rect = plt.Polygon(corners[:, [0, 2]], fill=None, edgecolor='r')
        ax.add_patch(rect)
    ax.set_xlim(xylim[0], xylim[1])
    ax.set_ylim(xylim[2], xylim[3])
    plt.savefig(save_path)
    print(f"Annotated BEV saved at {save_path}")


def visualize_2d_on_image(data_dict, save_path='vis_2d_on_image.jpg'):
    if save_path[-4:] != '.jpg' and save_path[-4:] != '.png':
        save_path = os.path.join(save_path, 'vis_2d_on_image.jpg')

    kitti_annotations = data_dict['annotations']
    image = cv2.imread(data_dict['image_path'])

    for seen, anno in zip(data_dict['seen'], kitti_annotations):
        name = anno[0]
        x1, y1, x2, y2 = anno[4:8]
        if seen:
            draw_2d_box(image, (x1, y1, x2, y2), color=(0, 255, 0))
            cv2.putText(image, name, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            draw_2d_box(image, (x1, y1, x2, y2), color=(0, 0, 255))
            cv2.putText(image, name, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(image, 'dataset: ' + data_dict['dataset'], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(image, 'seen object', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, 'unseen object', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite(save_path, image)
    print(f"Annotated image saved at {save_path}")


def visualize_3d_on_image(data_dict, save_path='vis_3d_on_image.jpg'):
    if save_path[-4:] != '.jpg' and save_path[-4:] != '.png':
        save_path = os.path.join(save_path, 'vis_3d_on_image.jpg')

    kitti_annotations = data_dict['annotations']
    image = cv2.imread(data_dict['image_path'])

    camera_internal = data_dict['camera_internal']
    cam_intrinsic = np.array([
        [camera_internal['fx'], 0, camera_internal['cx']],
        [0, camera_internal['fy'], camera_internal['cy']],
        [0, 0, 1]
    ])
    cam_intrinsic = np.hstack([cam_intrinsic, np.zeros((3, 1), dtype=np.float32)])

    for seen, anno in zip(data_dict['seen'], kitti_annotations):
        box_3d = Object3d(anno)
        corners_2d, corners_3d = compute_box_3d(box_3d, cam_intrinsic)
        if seen:
            draw_projected_box3d(image, corners_2d, color=(0, 255, 0))
        else:
            draw_projected_box3d(image, corners_2d, color=(0, 0, 255))

    cv2.putText(image, 'dataset: ' + data_dict['dataset'], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(image, 'seen object', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, 'unseen object', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite(save_path, image)
    print(f"Annotated image saved at {save_path}")
