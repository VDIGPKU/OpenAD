import os
import zipfile
from tqdm import tqdm
import json
import shutil
from .create_waymo import save_waymo
from .create_once import save_once
from .create_nuscenes import save_nuscenes
from .create_av2 import save_av2
from .create_kitti import save_kitti


def create_openad(
        av2_root='/path/to/argoverse2',
        kitti_root='/path/to/kitti_for_detection',
        nuscenes_root='/path/to/nuscenes',
        once_root='/path/to/once',
        waymo_root='/path/to/waymo',
        openad_zip='/path/to/openad.zip',
        openad_root='/path/to/save/openad'
):

    assert os.path.exists(os.path.join(av2_root, 'val'))
    assert os.path.exists(os.path.join(kitti_root, 'training'))
    assert os.path.exists(os.path.join(nuscenes_root, 'samples'))
    assert os.path.exists(os.path.join(once_root, 'data'))
    assert os.path.exists(os.path.join(waymo_root, 'validation'))

    for dir_name in ['camera_image', 'lidar_point_cloud', 'annotations', 'infos']:
        if os.path.exists(os.path.join(openad_root, dir_name)):
            print(f"The directory '{os.path.join(openad_root, dir_name)}' has been deleted.")
            shutil.rmtree(os.path.join(openad_root, dir_name))

    os.mkdir(os.path.join(openad_root, 'camera_image'))
    os.mkdir(os.path.join(openad_root, 'lidar_point_cloud'))

    zip_file = zipfile.ZipFile(openad_zip)
    zip_list = zip_file.namelist()
    for f in zip_list:
        zip_file.extract(f, openad_root)
    zip_file.close()

    once_dataset = None

    for idx in tqdm(range(250 + 309 + 134 + 1057, 2000)):
        '''
            av2       range(0, 250)
            kitti     range(250, 250 + 309)
            nuscenes  range(250 + 309, 250 + 309 + 134)
            once      range(250 + 309 + 134, 250 + 309 + 134 + 1057)
            waymo     range(250 + 309 + 134 + 1057, 2000)
        '''
        with open(os.path.join(openad_root, 'infos', str(idx) + '.json'), 'r', encoding='utf-8') as file:
            info = json.load(file)

        dataset = info["dataset"]

        if dataset == 'waymo':
            sensors_path = info["sensors_path"]
            save_waymo(sensors_path, waymo_root, idx, openad_root)
        else:
            img_path_list = info["img_path"]
            pc_path_list = info["lidar_path"]
            if dataset == 'once':
                once_dataset = save_once(once_dataset, once_root, img_path_list, pc_path_list, openad_root, idx)
            elif dataset == 'nuscenes':
                save_nuscenes(nuscenes_root, img_path_list, pc_path_list, openad_root, idx)
            elif dataset == 'av2':
                save_av2(av2_root, img_path_list, pc_path_list, openad_root, idx)
            elif dataset == 'kitti':
                save_kitti(kitti_root, img_path_list, pc_path_list, openad_root, idx)
