import torch
import torch.nn as nn
import numpy as np
from robosuite.utils.transform_utils import quat_distance, quat2axisangle


DISTANCE_METRICS = {"l1", "l2", "linf", "combined"}
POSE_LOSS_MODES = {"position", "pose", "val"}


class PoseDistanceLoss(nn.Module):
    """
    Simple class to compute pose distance (position distance + orientation distance)
    """
    def __init__(self, distance_metric="l2", scale_factor=1.0, alpha=1.0, epsilon=1e-4, mode="pose"):
        """
        Initializes this custom loss for pose distance

        Args:
            distance_metric (str): Distance metric to use. Options are l1, l2, linf, or combined
            scale_factor (float): Scaling factor for final loss value
            alpha (float): Scaling factor to weight orientation error relative to position error
            epsilon (float): Tolerance to add to sqrt to prevent nan's from occurring (sqrt(0))
            mode (str): Mode to run this loss criterion -- options are as follows:
                "position" : only position error is used
                "pose" : full pose (position + orientation) error is used
                "val" : validation mode. Outputs the summed L2 position error and radian angle error
        """
        # Always run superclass init first
        super(PoseDistanceLoss, self).__init__()

        # Set distance_metric, scale value, epsilon value, and position only flag
        if distance_metric in DISTANCE_METRICS:
            self.distance_metric = distance_metric
        else:
            raise ValueError("Invalid distance metric specified; available are: {}, requested {}.".format(
                DISTANCE_METRICS, distance_metric))
        self.scale_factor = scale_factor
        self.alpha = alpha
        self.epsilon = epsilon
        if mode in POSE_LOSS_MODES:
            self.mode = mode
        else:
            raise ValueError("Invalid loss mode specified; available are: {}, requested {}.".format(
                POSE_LOSS_MODES, mode))

    def forward(self, prediction, truth):
        """
        Computes position distance + orientation distance (squared) between prediction and orientation if
            self.position_only is set to False, else only computes position distance.
            Assumes prediction and truth are of the same size

        Args:
            prediction (torch.Tensor): Prediction of estimated pose (*, 7) of form (x,y,z,i,j,k,w)
                Note: Assumes UNNORMALIZED quat part
            truth (torch.Tensor): Ground truth state of pose, of shape (*, 7) of form (x,y,z,i,j,k,w)
                Note: Assumes NORMALIZED quat part with POSITIVE w value

        Returns:
            distance (torch.Tensor): (Summed) distances across all inputs. Shape = (1,)
        """

        # First, split tensor into separate parts
        predict_pos, predict_ori = torch.split(prediction, (3,4), dim=-1)
        true_pos, true_ori = torch.split(truth, (3, 4), dim=-1)

        # Normalize the predicted ori
        quat_mag = torch.sqrt(torch.sum(predict_ori.pow(2), dim=-1, keepdim=True))      # Shape (*,1)
        predict_ori = predict_ori / quat_mag

        # Compute position loss based on distance metric specified
        if self.distance_metric == "l2":
            # Compute pos error (cartesian distance)
            summed_pos_sq = torch.sum((predict_pos - true_pos).pow(2), dim=-1, keepdim=False)        # Shape (*)
            pos_dist = torch.sum(torch.sqrt(summed_pos_sq + self.epsilon))
        elif self.distance_metric == "l1" or self.distance_metric == "linf":
            # Compute absolute values first
            dist_pos_abs = torch.abs(predict_pos - true_pos)
            if self.distance_metric == "l1":
                pos_dist = torch.sum(dist_pos_abs)
            else:   # linf case
                pos_dist = torch.sum(torch.max(dist_pos_abs, dim=-1)[0])
        else:   #combined case
            # Compute l2 loss
            summed_pos_sq = torch.sum((predict_pos - true_pos).pow(2), dim=-1, keepdim=False)  # Shape (*)
            l2_pos_dist = torch.sum(torch.sqrt(summed_pos_sq + self.epsilon))
            # Compute l1 and linf losses
            dist_pos_abs = torch.abs(predict_pos - true_pos)
            l1_pos_dist = torch.sum(dist_pos_abs)
            linf_pos_dist = torch.sum(torch.max(dist_pos_abs, dim=-1)[0])
            # Combine distances
            pos_dist = l2_pos_dist + l1_pos_dist + linf_pos_dist

        # Additionally compute orientation error if requested
        if self.mode == "val":
            # We directly compute the summed absolute angle error in radians
            # Note: we don't care about torch-compatible computations since we're not backpropping in this case
            # Convert torch tensors to numpy arrays
            predict_ori = predict_ori.view(-1, 4).cpu().detach().numpy()
            true_ori = true_ori.view(-1, 4).cpu().detach().numpy()
            # Create var to hold summed orientation error
            summed_angle_err = 0
            # Loop over all quats and compute angle error
            for i in range(predict_ori.shape[0]):
                _, angle = quat2axisangle(quat_distance(predict_ori[i], true_ori[i]))
                # Confine the error to be between +/- pi
                if angle < -np.pi:
                    angle += 2*np.pi
                elif angle > np.pi:
                    angle -= 2*np.pi
                summed_angle_err += abs(angle)
            # Immediately return the position and orientation errors
            return pos_dist.cpu().detach().numpy(), summed_angle_err
        elif self.mode == "pose":
            # Compute ori error (quat distance), roughly d(q1,q2) = 1 - <q1,q2>^2 where <> is the inner product
            # see https://math.stackexchange.com/questions/90081/quaternion-distance
            inner_product = torch.sum(predict_ori * true_ori, dim=-1, keepdim=False)    # Shape (*)
            ori_dist_squared = torch.sum(1 - inner_product.pow(2))
            # We also add in penalty if w component is negative
            # see https://dspace.mit.edu/bitstream/handle/1721.1/119699/1078151125-MIT.pdf?sequence=1
            w_penalty = torch.clamp(-predict_ori[..., -1], min=0)
            ori_dist_squared += torch.sum(w_penalty)
        else:   # case "position"
            # No orientation loss used
            ori_dist_squared = 0

        # Return summed dist
        return self.scale_factor * (pos_dist + self.alpha * ori_dist_squared)
