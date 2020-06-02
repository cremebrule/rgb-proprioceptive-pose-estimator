import torch
import torch.nn as nn


DISTANCE_METRICS = {"l1", "l2", "linf"}


class PoseDistanceLoss(nn.Module):
    """
    Simple class to compute pose distance (position distance + orientation distance)
    """
    def __init__(self, distance_metric="l2", scale_factor=1.0, epsilon=1e-4, position_only=True):
        """
        Initializes this custom loss for pose distance

        Args:
            distance_metric (str): Distance metric to use. Options are l1, l2, and linf
            scale_factor (float): Scaling factor for final loss value
            epsilon (float): Tolerance to add to sqrt to prevent nan's from occurring (sqrt(0))
            position_only (bool): Whether to train for both position and orientation error or just position
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
        self.epsilon = epsilon
        self.position_only = position_only

    def forward(self, prediction, truth):
        """
        Computes position distance + orientation distance (squared) between prediction and orientation if
            self.position_only is set to False, else only computes position distance.
            Assumes prediction and truth are of the same size

        Args:
            prediction (torch.Tensor): Prediction of estimated pose (*, 7)
                Note: Assumes UNNORMALIZED quat part
            truth (torch.Tensor): Ground truth state of pose, of shape (*, 7)
                Note: Assumes NORMALIZED quat part

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
        else: # l1, linf case
            # Compute absolute values first
            dist_pos_abs = torch.abs(predict_pos - true_pos)
            if self.distance_metric == "l1":
                pos_dist = torch.sum(dist_pos_abs)
            else:   # linf case
                pos_dist = torch.sum(torch.max(dist_pos_abs, dim=-1)[0])

        if not self.position_only:
            # Compute ori error (quat distance) - angle = acos(2<q1,q2>^2 - 1)
            inner_product = torch.sum(predict_ori * true_ori, dim=-1, keepdim=False)    # Shape (*)
            ori_dist_squared = torch.sum((torch.acos(2 * inner_product.pow(2) - 1)).pow(2))
        else:
            ori_dist_squared = 0

        # Return summed dist
        #print(summed_pos_sq)
        #print(pos_dist)
        #print(inner_product)
        #print(ori_dist_squared)
        return self.scale_factor * (pos_dist + ori_dist_squared)
