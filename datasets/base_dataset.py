"""
定义各类DATASET类
"""

from datasets.transforms import get_transforms



def construct_dataset(cfg, mode, **kwargs):
    """
    This function constructs datasets.
    """
    # dataset arguments for the dataloader
    if mode == 'train':
        dataset_args = {
            'cameras': cfg['data']['cameras'],   # ['camera_01', 'camera_05', 'camera_06', 'camera_07', 'camera_08', 'camera_09']
            'back_context': cfg['data']['back_context'],  # 1
            'forward_context': cfg['data']['forward_context'],  # 1
            'data_transform': get_transforms('train', **kwargs),  
            'depth_type': cfg['data']['depth_type'] if 'gt_depth' in cfg['data']['train_requirements'] else None,  # 如果训练时需要用gt_depth，则depth_type=lidar,否则为None
            'scale_range': cfg['model']['fusion_level'] if 'fusion_level' in cfg['model'] else -1,  # 如果使用"fusion"类型网络，则scale_range=2，否则为-1
            'with_pose': 'gt_pose' in cfg['data']['train_requirements'],  # 训练时是否使用pose_net
            'with_mask': 'mask' in cfg['data']['train_requirements']  # 训练时是否使用mask
        }
        
    elif mode == 'val':
        dataset_args = {
            'cameras': cfg['data']['cameras'],
            'back_context': cfg['data']['back_context'],
            'forward_context': cfg['data']['forward_context'],
            'data_transform': get_transforms('train', **kwargs), # for aligning inputs without any augmentations
            'depth_type': cfg['data']['depth_type'] if 'gt_depth' in cfg['data']['val_requirements'] else None,
            'scale_range': cfg['model']['fusion_level'] if 'fusion_level' in cfg['model'] else -1,
            'with_pose': 'gt_pose' in cfg['data']['val_requirements'],
            'with_mask': 'mask' in cfg['data']['val_requirements']            
        }

    elif mode == "pose":
        dataset_args = {
            'cameras': cfg['data']['cameras'],
            'back_context': cfg['data']['back_context'],
            'forward_context': cfg['data']['forward_context'],
            'data_transform': get_transforms('train', **kwargs),
            'depth_type': None,
            'scale_range': -1,
            'with_pose': 'gt_pose' in cfg['data']['val_requirements'],
            'with_mask': 'mask' in cfg['data']['val_requirements']    
        }
        
    # DDAD dataset
    if cfg['data']['dataset'] == 'ddad':
        from datasets.ddad_dataset import DDADDataset
        if mode == "pose":
            mode = 'train'
        dataset = DDADDataset(
            cfg['data']['data_path'], mode,
            **dataset_args
        )
    elif cfg['data']['dataset'] == 'nuscenes':
        from datasets.nuscenes_dataset import NuScenesdataset 
        dataset = NuScenesdataset(
            cfg['data']['data_path'], mode,
            **dataset_args
        )       
    else:
        raise ValueError('Unknown dataset: ' + cfg['data']['dataset'])
    return dataset