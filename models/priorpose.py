import torch
from geometry.pose import Pose, vec_to_matrix
from models import PoseModelBase
from networks import PriorPoseNet


class PriorPoseModel(PoseModelBase):
    def __init__(self, cfg, rank):
        super().__init__(cfg, rank)
        net = PriorPoseNet(cfg).to(rank)
        if self.ddp_enable:
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank])
        self.models['prior_pose'] = net
        self.pose_helper = Pose(cfg)

    def process_batch(self, inputs, rank):
        inputs = self._to_device(inputs, rank)
        full_gt = self.pose_helper.compute_pose_from_gt(inputs)
        gt_cam0 = full_gt[('cam', 0)]

        net = self.models['prior_pose']
        raw_net = net.module if hasattr(net, 'module') else net

        pred_cam0 = {}
        for f_i in self.frame_ids[1:]:
            frame_ids = [-1, 0] if f_i < 0 else [0, 1]
            result = raw_net(inputs, frame_ids, cam=0)
            
            if isinstance(result, tuple):
                axisangle, translation = result
                pred_cam0[('cam_T_cam', 0, f_i)] = vec_to_matrix(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
            else:
                pred_cam0[('cam_T_cam', 0, f_i)] = result

        losses = self.pose_loss(pred_cam0, gt_cam0)
        return losses, pred_cam0, gt_cam0