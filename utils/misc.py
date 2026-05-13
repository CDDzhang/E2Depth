# Copyright 2020 Toyota Research Institute.  All rights reserved.
import yaml
import os
from collections import defaultdict
import torch

_NUSC_CAM_LIST = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']
_DDAD_CAM_LIST = ['camera_01', 'camera_05', 'camera_06', 'camera_07', 'camera_08', 'camera_09']
_REL_CAM_DICT = {0: [1,2], 1: [0,3], 2: [0,4], 3: [1,5], 4: [2,5], 5: [3,4]}

def get_proj_dir():
    # This function returns the project root directory
    cur_path = os.path.dirname(os.path.realpath(__file__))
    proj_path = os.path.dirname(cur_path)
    return proj_path

def load_cfg(config_file, mode='train'):
    proj_dir = get_proj_dir()
    config_file = os.path.join(proj_dir, "configs", config_file)
    cfg = get_config(config_file, mode='train')
    return cfg

def filter_dict(dictionary, keywords):
    # 获取一个字典里所存在的所有key，并返回一个list
    return [key for key in keywords if key in dictionary]

def make_list(var, n=None):
    # 把input的变量变换成list， 如果数量不够则repeats到n
    var = var if  isinstance(var, list) else [var]
    if n is None:
        return var
    else:
        assert len(var) == 1 or len(var) == n, 'Wrong list length for make_list'
        return var * n if len(var) == 1 else var

def same_shape(shape1, shape2):
    # 判断两个输入的len是否一致
    if len(shape1) != len(shape2):
        return False
    for i in range(len(shape1)):
        if shape1[i] != shape2[i]:
            return False
    return True

def parse_crop_borders(borders, shape):
    """
    Calculate borders for cropping.

    Parameters
    ----------
    borders : tuple
        Border input for parsing. Can be one of the following forms:
        (int, int, int, int): y, height, x, width
        (int, int): y, x --> y, height = image_height - y, x, width = image_width - x
        Negative numbers are taken from image borders, according to the shape argument
        Float numbers for y and x are treated as percentage, according to the shape argument,
            and in this case height and width are centered at that point.
    shape : tuple
        Image shape (image_height, image_width), used to determine negative crop boundaries

    Returns
    -------
    borders : tuple (left, top, right, bottom)
        Parsed borders for cropping
    """
    # Return full image if there are no borders to crop
    if len(borders) == 0:
        return 0, 0, shape[1], shape[0]
    # Copy borders for modification
    borders = list(borders).copy()
    # If borders are 4-dimensional
    if len(borders) == 4:
        borders = [borders[2], borders[0], borders[3], borders[1]]
        if isinstance(borders[0], int):
            # If horizontal cropping is integer (regular cropping)
            borders[0] += shape[1] if borders[0] < 0 else 0
            borders[2] += shape[1] if borders[2] <= 0 else borders[0]
        else:
            # If horizontal cropping is float (center cropping)
            center_w, half_w = borders[0] * shape[1], borders[2] / 2
            borders[0] = int(center_w - half_w)
            borders[2] = int(center_w + half_w)
        if isinstance(borders[1], int):
            # If vertical cropping is integer (regular cropping)
            borders[1] += shape[0] if borders[1] < 0 else 0
            borders[3] += shape[0] if borders[3] <= 0 else borders[1]
        else:
            # If vertical cropping is float (center cropping)
            center_h, half_h = borders[1] * shape[0], borders[3] / 2
            borders[1] = int(center_h - half_h)
            borders[3] = int(center_h + half_h)
    # If borders are 2-dimensional
    elif len(borders) == 2:
        borders = [borders[1], borders[0]]
        if isinstance(borders[0], int):
            # If cropping is integer (regular cropping)
            borders = (max(0, borders[0]),
                       max(0, borders[1]),
                       shape[1] + min(0, borders[0]),
                       shape[0] + min(0, borders[1]))
        else:
            # If cropping is float (center cropping)
            center_w, half_w = borders[0] * shape[1], borders[1] / 2
            center_h, half_h = borders[0] * shape[0], borders[1] / 2
            borders = (int(center_w - half_w), int(center_h - half_h),
                       int(center_w + half_w), int(center_h + half_h))
    # Otherwise, invalid
    else:
        raise NotImplementedError('Crop tuple must have 2 or 4 values.')
    # Assert that borders are valid
    assert 0 <= borders[0] < borders[2] <= shape[1] and \
           0 <= borders[1] < borders[3] <= shape[0], 'Crop borders {} are invalid'.format(borders)
    # Return updated borders
    return borders

def pretty_ts(ts):
    """
    This function prints amount of time taken in user friendly way.
    """
    second = int(ts)
    minute = second // 60
    hour = minute // 60
    return f'{hour:02d}h{(minute%60):02d}m{(second%60):02d}s'



def camera2ind(cameras):
    """
    This function transforms camera name list to indices 
    """    
    indices = []
    for cam in cameras:
        if cam in _DDAD_CAM_LIST:
            ind = _DDAD_CAM_LIST.index(cam)
        elif cam in _NUSC_CAM_LIST:
            ind = _NUSC_CAM_LIST.index(cam)
        else:
            ind = None
        indices.append(ind)
    return indices


def get_relcam(cameras):
    """
    This function returns relative camera indices from given camera list
    """
    relcam_dict = defaultdict(list)
    indices = camera2ind(cameras)
    for ind in indices:
        relcam_dict[ind] = []
        relcam_cand = _REL_CAM_DICT[ind]
        for cand in relcam_cand:
            if cand in indices:
                relcam_dict[ind].append(cand)
    return relcam_dict        

def get_config(config, mode='train', weight_path=None):
    """
    This function reads the configuration file and return as dictionary
    """
    with open(config, 'r') as stream:
        cfg = yaml.load(stream, Loader=yaml.FullLoader)

        cfg_name = os.path.splitext(os.path.basename(config))[0]
        print('Experiment: ', cfg_name)

        _log_path = os.path.join(cfg['data']['log_dir'], cfg_name)
        cfg['data']['log_path'] = _log_path
        cfg['data']['save_weights_root'] = os.path.join(_log_path, 'models')
        cfg['data']['num_cams'] = len(cfg['data']['cameras'])
        cfg['model']['mode'] = mode
        cfg['data']['rel_cam_list'] = get_relcam(cfg['data']['cameras'])
        
        if mode == 'train':
            cfg['eval']['syn_visualize'] = False # for pretrained 
            
        elif mode == 'eval':
            cfg['ddp']['world_size'] = 1
            cfg['ddp']['gpus'] = [0]
            cfg['training']['batch_size'] = cfg['eval']['eval_batch_size']
            cfg['training']['depth_flip'] = False
    return cfg