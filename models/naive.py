import torch
import torch.nn as nn
import torch.nn.functional as F
from util.model_utils import import_resnet


class NaiveEndEffectorStateEstimator(nn.Module):
    """
    A naive estimator for determining another arm's eef state based on visual observation and active arm's (noisy)
    measurement of its own state. Does not take into account any temporal dependencies (i.e.: this is a one-shot
    regressor)
    """

    def __init__(
            self,
            hidden_dims_pre_measurement,
            hidden_dims_post_measurement,
            num_resnet_layers=50,
            latent_dim=50,
            feature_extract=True,
    ):
        """
        Args:
            hidden_dims_pre_measurement (list of ints): Number of FC + highway hidden layers to include before the
                self measurement is taken into accout

            hidden_dims_post_measurement (list of ints): Number of FC + highway hidden layers to include after the
                self measurement is taken into accout

            num_resnet_layers (int): Number of layers for imported, pretrained ResNet model.
                Options are 18, 34, 50, 101, 152

            latent_dim (int): Latent space dimension size; this is the output of the ResNet network

            feature_extract (bool): Whether we're feature extracting from ResNet or finetuning
        """
        # Always run super init first
        super(NaiveEndEffectorStateEstimator, self).__init__()

        # Import ResNet as feature model
        self.feature_net, _ = import_resnet(num_resnet_layers, latent_dim, feature_extract)

        # TODO: Need to initialize weights for FC layers!

        # Define FC layers to run
        pre_hidden_dims = [latent_dim] + hidden_dims_pre_measurement + [7]
        for i, hidden_dim in enumerate(pre_hidden_dims):
            if i == len(pre_hidden_dims) - 1:
                continue
            else:
                setattr(self, "pre_fc{}".format(i), nn.Linear(hidden_dim, pre_hidden_dims[i+1]))

        self.n_pre_hidden = len(pre_hidden_dims) - 1

        post_hidden_dims = [latent_dim + 7] + hidden_dims_post_measurement + [7]
        for i, hidden_dim in enumerate(post_hidden_dims):
            if i == len(post_hidden_dims) - 1:
                continue
            else:
                setattr(self, "post_fc{}".format(i), nn.Linear(hidden_dim, post_hidden_dims[i + 1]))

        self.n_post_hidden = len(post_hidden_dims) - 1

        # Set rollout to false by default
        self.rollout = False

    def forward(self, img, depth, self_measurement):
        """
        Forward pass for this model

        Args:
            img (torch.Tensor): tensor representing batch of images of shape (N, C, H, W)
            depth (torch.Tensor): tensor representing batch of depth images of shape (N, 1, H, W)
            self_measurement (torch.Tensor): tensor representing batch of measurements of active robot's eef state
                of shape (N, 7)

        Returns:
            pre_out (torch.Tensor): output from pre-measurement branch of forward pass, of shape (N, 7)
            post_out (torch.Tensor): output from post-measurement branch of forward pass, of shape (N, 7)
        """
        # TODO: Check to make sure ResNet is in same eval() or train() mode as top level layers
        # First, pass img through ResNet to extract features
        features = self.feature_net(img)

        # Copy these features to pass through pre-measurement layers
        pre_out = features.clone()

        # Pass input through FC + Highway layers pre-measurement
        for i in range(self.n_pre_hidden):
            # Pass through FC + Activation
            pre_out = F.relu(getattr(self, "pre_fc{}".format(i))(pre_out))

            # Add in highway layer
            # TODO: Not done right now

        # Evaluate difference between self measurement and pre_out
        measurement_diff = pre_out - self_measurement

        # Concatenate features with diff measurement
        post_out = torch.cat([features, measurement_diff], dim=1)

        # Pass input through FC + Highway layers post-measurement
        for i in range(self.n_post_hidden):
            # Pass through FC + Activation
            post_out = F.relu(getattr(self, "post_fc{}".format(i))(post_out))

            # Add in highway layer
            # TODO: Not done right now

        # Return final output
        return pre_out, post_out

    def reset_initial_state(self, batch_size):
        """
        Resets any initial state. Doesn't actually do anything for this class since it doesn't have any temporal
        dependencies


        Args:
            batch_size (int): Batch size currently being run
        """
        pass

    @property
    def requires_sequence(self):
        return False
