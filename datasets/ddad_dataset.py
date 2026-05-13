import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from dgp.datasets import SynchronizedSceneDataset
from dgp.utils.camera import Camera, generate_depth_map
from dgp.utils.pose import Pose
from utils.misc import make_list, get_config
from datasets.dgp_dataset import stack_sample, DGPDataset
from datasets.data_utils import transform_mask_sample, img_loader, mask_loader_scene, align_dataset
from datasets.transforms import get_transforms


class DDADDataset(DGPDataset):
    def __init__(self, *args, with_mask, scale_range, **kwargs):
        # *args是原来就有的参数，**kwargs是新增的参数
        super().__init__(*args, **kwargs)
        self.cameras = kwargs['cameras']  # [Camera_01, Camera_05, Camera_06, Camera_07, Camera_08, Camera_09]
        self.scales = np.arange(scale_range+2)  # fusion_level+3=5 # e.g., [0, 1, 2, 3, 4] for scale_range=2

        ## self-occ masks 
        self.with_mask = with_mask
        if self.with_mask:
            cur_path = os.path.dirname(os.path.realpath(__file__))
            self.mask_path = os.path.join(cur_path, 'ddad_mask')
            file_name = os.path.join(self.mask_path, 'mask_idx_dict.pkl')
            self.mask_idx_dict = pd.read_pickle(file_name)
            self.mask_loader = mask_loader_scene

        datum_names = self.cameras + ['lidar']

        # 此时读取的pose值已经默认转换为了Pose类，可以直接使用其进行计算
        self.dataset = SynchronizedSceneDataset(self.path,
                        split=self.split,
                        datum_names=datum_names,
                        backward_context=self.bwd,
                        forward_context=self.fwd,
                        requested_annotations=None,
                        only_annotated_datums=False)             
    
    def __getitem__(self, idx):
        # get DGP sample (if single sensor, make it a list)
        self.sample_dgp = self.dataset[idx]
        self.sample_dgp = [make_list(sample) for sample in self.sample_dgp]
        sample = []
        contexts = []
        if self.bwd:
            contexts.append(-1)
        if self.fwd:
            contexts.append(1)
        # for self-occ mask
        scene_idx, sample_idx_in_scene, datum_indices = self.dataset.dataset_item_index[idx]  
        scene_dir = self.dataset.scenes[scene_idx].directory
        scene_name = os.path.basename(scene_dir)  # '000002'
        if self.with_mask:
            mask_idx = self.mask_idx_dict[int(scene_name)]  # 2
        
        # loop over all cameras
        for cam in range(self.num_cameras):
            scene_dir = self.dataset.scenes[scene_idx].directory
            filename = self.dataset.get_datum(
                scene_idx, sample_idx_in_scene, datum_indices[cam]).datum.image.filename
            # rgb/CAMERA_01/15593298273460162.png
            # 000002/{}/CAMERA_01/15593298273460162  # 去掉了拓展名
            filename =  os.path.splitext(os.path.join(os.path.basename(scene_dir), filename.replace('rgb', '{}')))[0]

            data = {
                'idx': idx,
                'dataset_idx': self.dataset_idx,
                'sensor_name': self.get_current('datum_name', cam), # get current只会拿到名字，不会拿到数据
                'contexts': contexts, # [-1, 1]
                'filename': filename,
                'splitname': '%s_%010d' % (self.split, idx),                
                'rgb': self.get_current('rgb', cam), # 原图，没必要
                'intrinsics': self.get_current('intrinsics', cam), # 内参，必须读取
            }  

            # if depth is returned
            if self.with_depth:
                depth_filename = '{}/{}.npz'.format(os.path.dirname(self.path), filename.format('depth'))
                # load and return if exists
                depth = np.load(depth_filename, allow_pickle=True)['depth'] if os.path.exists(depth_filename) else None
                data.update({'depth': depth})
            # if pose is returned
            if self.with_pose:
                data.update({'extrinsics': self.get_current('extrinsics', cam).matrix,
                             'pose': self.get_current('pose', cam).matrix})  # 只保留了一个4*4的矩阵
                
            # with mask
            if self.with_mask:
                data.update({'mask': self.mask_loader(self.mask_path, mask_idx, self.cameras[cam])})
            # 同时更新上下文的值
            if self.has_context:
                data.update({'rgb_context': self.get_context('rgb', cam)})
                if self.with_pose:
                    pose_context = self.get_context('pose', cam)
                    data.update({
                        # 其实外参是完全一致的，不需要怎么变化
                        'pose_bwd': pose_context[0].matrix if self.bwd else None,
                        'pose_fwd': pose_context[1].matrix if self.fwd and self.bwd else (
                                     pose_context[0].matrix if self.fwd and not self.bwd else None)
                    })
            sample.append(data) 
            
        
        if self.data_transform:
            sample = [self.data_transform(smp) for smp in sample]
            if self.with_mask:
                sample = [transform_mask_sample(smp, self.data_transform) for smp in sample]
        
        # stack and align dataset for our trainer
        sample = stack_sample(sample)
        sample = align_dataset(sample, self.scales, contexts, with_pose=True)
        return sample 