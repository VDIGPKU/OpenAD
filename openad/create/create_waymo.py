import os
import zipfile
from tqdm import tqdm
import json
import shutil
import numpy as np
import tensorflow as tf
from waymo_open_dataset.utils import range_image_utils, transform_utils
from waymo_open_dataset.utils.frame_utils import parse_range_image_and_camera_projection
from waymo_open_dataset import dataset_pb2
import open3d as o3d
import cv2


def save_waymo(sensors_path, waymo_root, idx, openad_root):
    for frame_idx, item in enumerate(reversed(sensors_path)):
        data = tf.data.TFRecordDataset(os.path.join(waymo_root, item[0]), compression_type='')
        for frame_iii, data_i in enumerate(data):
            if frame_iii == item[1]:
                frame = dataset_pb2.Frame()
                frame.ParseFromString(bytearray(data_i.numpy()))

                img = frame.images[0]
                img_path = f'camera_image/{idx}_{frame_idx}.jpg'
                img = imfrombytes(img.image)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(img_path, img)

                save_lidar_path = os.path.join(openad_root, 'lidar_point_cloud', f'{idx}_{frame_idx}.pcd')
                save_lidar(frame, save_lidar_path)


def imfrombytes(content):
    img_np = np.frombuffer(content, np.uint8)
    flag = cv2.IMREAD_COLOR
    img = cv2.imdecode(img_np, flag)
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    return img


def save_lidar(frame, save_path):
    """Parse and save the lidar data in psd format.
    """
    range_images, camera_projections, range_image_top_pose = \
        parse_range_image_and_camera_projection(frame)

    # First return
    points_0, cp_points_0, intensity_0, elongation_0, mask_indices_0 = \
        convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            ri_index=0
        )
    points_0 = np.concatenate(points_0, axis=0)
    intensity_0 = np.concatenate(intensity_0, axis=0)
    elongation_0 = np.concatenate(elongation_0, axis=0)
    mask_indices_0 = np.concatenate(mask_indices_0, axis=0)

    # Second return
    points_1, cp_points_1, intensity_1, elongation_1, mask_indices_1 = \
        convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            ri_index=1
        )
    points_1 = np.concatenate(points_1, axis=0)
    intensity_1 = np.concatenate(intensity_1, axis=0)
    elongation_1 = np.concatenate(elongation_1, axis=0)
    mask_indices_1 = np.concatenate(mask_indices_1, axis=0)

    points = np.concatenate([points_0, points_1], axis=0)
    intensity = np.concatenate([intensity_0, intensity_1], axis=0)
    elongation = np.concatenate([elongation_0, elongation_1], axis=0)
    mask_indices = np.concatenate([mask_indices_0, mask_indices_1], axis=0)

    # timestamp = frame.timestamp_micros * np.ones_like(intensity)

    # concatenate x,y,z, intensity, elongation, timestamp (6-dim)
    point_cloud = np.column_stack(
        (points, intensity, elongation, mask_indices))

    # print(point_cloud.shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud.astype(np.float32)[:, :3])
    o3d.io.write_point_cloud(save_path, pcd, write_ascii=True)


def convert_range_image_to_point_cloud(frame,
                                       range_images,
                                       camera_projections,
                                       range_image_top_pose,
                                       ri_index=0):
    calibrations = sorted(
        frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    cp_points = []
    intensity = []
    elongation = []
    mask_indices = []

    frame_pose = tf.convert_to_tensor(
        value=np.reshape(np.array(frame.pose.transform), [4, 4]))
    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(value=range_image_top_pose.data),
        range_image_top_pose.shape.dims)
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = \
        transform_utils.get_rotation_matrix(
            range_image_top_pose_tensor[..., 0],
            range_image_top_pose_tensor[..., 1],
            range_image_top_pose_tensor[..., 2])
    range_image_top_pose_tensor_translation = \
        range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation)
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        if len(c.beam_inclinations) == 0:
            beam_inclinations = range_image_utils.compute_inclination(
                tf.constant(
                    [c.beam_inclination_min, c.beam_inclination_max]),
                height=range_image.shape.dims[0])
        else:
            beam_inclinations = tf.constant(c.beam_inclinations)

        beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
        extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image.data),
            range_image.shape.dims)
        pixel_pose_local = None
        frame_pose_local = None
        if c.name == dataset_pb2.LaserName.TOP:
            pixel_pose_local = range_image_top_pose_tensor
            pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
            frame_pose_local = tf.expand_dims(frame_pose, axis=0)
        range_image_mask = range_image_tensor[..., 0] > 0

        range_image_cartesian = \
            range_image_utils.extract_point_cloud_from_range_image(
                tf.expand_dims(range_image_tensor[..., 0], axis=0),
                tf.expand_dims(extrinsic, axis=0),
                tf.expand_dims(tf.convert_to_tensor(
                    value=beam_inclinations), axis=0),
                pixel_pose=pixel_pose_local,
                frame_pose=frame_pose_local)

        mask_index = tf.where(range_image_mask)

        range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
        points_tensor = tf.gather_nd(range_image_cartesian, mask_index)

        cp = camera_projections[c.name][ri_index]
        cp_tensor = tf.reshape(
            tf.convert_to_tensor(value=cp.data), cp.shape.dims)
        cp_points_tensor = tf.gather_nd(cp_tensor, mask_index)
        points.append(points_tensor.numpy())
        cp_points.append(cp_points_tensor.numpy())

        intensity_tensor = tf.gather_nd(range_image_tensor[..., 1],
                                        mask_index)
        intensity.append(intensity_tensor.numpy())

        elongation_tensor = tf.gather_nd(range_image_tensor[..., 2],
                                         mask_index)
        elongation.append(elongation_tensor.numpy())
        if c.name == 1:
            mask_index = (ri_index * range_image_mask.shape[0] +
                          mask_index[:, 0]
                          ) * range_image_mask.shape[1] + mask_index[:, 1]
            mask_index = mask_index.numpy().astype(elongation[-1].dtype)
        else:
            mask_index = np.full_like(elongation[-1], -1)

        mask_indices.append(mask_index)

    return points, cp_points, intensity, elongation, mask_indices

