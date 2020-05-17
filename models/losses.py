import torch
import torch.nn as nn


class PoseDistanceLoss(nn.Module):
    """
    Simple class to compute pose distance (position distance + orientation distance)
    """
    def __init__(self, epsilon=1e-4, position_only=True):
        """
        Initializes this custom loss for pose distance

        Args:
            epsilon (float): Tolerance to add to sqrt to prevent nan's from occurring (sqrt(0))
            position_only (bool): Whether to train for both position and orientation error or just position
        """
        # Always run superclass init first
        super(PoseDistanceLoss, self).__init__()

        # Set epsilon value and position only flag
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

        # Compute pos error (cartesian distance)
        summed_pos_sq = torch.sum((predict_pos - true_pos).pow(2), dim=-1, keepdim=False)        # Shape (*)
        pos_dist = torch.sum(torch.sqrt(summed_pos_sq + self.epsilon))

        if not self.position_only:
            # Compute ori error (quat distance) - angle = acos(2<q1,q2>^2 - 1)
            inner_product = torch.sum(predict_ori * true_ori, dim=-1, keepdim=False)    # Shape (*)
            ori_dist_squared = torch.sum((torch.acos(2 * inner_product.pow(2) - 1)).pow(2))
        else:
            ori_dist_squared = 0

        # Return summed dist
        print(summed_pos_sq)
        print(pos_dist)
        #print(inner_product)
        #print(ori_dist_squared)
        return pos_dist + ori_dist_squared
