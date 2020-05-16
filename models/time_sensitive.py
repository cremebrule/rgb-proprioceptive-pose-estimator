import torch
import torch.nn as nn
import torch.nn.functional as F
from util.model_utils import import_resnet


class TemporallyDependentStateEstimator(nn.Module):
    """
    A less-naive estimator for determining another arm's eef state based on visual observation and active arm's (noisy)
    measurement of its own state. Takes into account temporal dependencies via LSTM module
    """

    def __init__(
            self,
            hidden_dim_pre_measurement,
            hidden_dim_post_measurement,
            num_resnet_layers=50,
            latent_dim=50,
            sequence_length=10,
    ):
        """
        Args:
            hidden_dim_pre_measurement (int): size of hidden state for LSTM before the
                self measurement is taken into account

            hidden_dim_post_measurement (int): size of hidden state for LSTM after the
                self measurement is taken into account

            num_resnet_layers (int): Number of layers for imported, pretrained ResNet model.
                Options are 18, 34, 50, 101, 152

            latent_dim (int): Latent space dimension size; this is the output of the ResNet network

            sequence_length (int): Size of sequences to be input into LSTM
        """
        # Always run super init first
        super(TemporallyDependentStateEstimator, self).__init__()

        # Import ResNet as feature model
        self.feature_net, _ = import_resnet(num_resnet_layers, latent_dim)

        # Define LSTM nets
        self.pre_measurement_rnn = nn.LSTMCell(input_size=latent_dim, hidden_size=hidden_dim_pre_measurement)
        self.pre_measurement_fc = nn.Linear(hidden_dim_pre_measurement, 7)
        self.post_measurement_rnn = nn.LSTMCell(input_size=latent_dim + 7, hidden_size=hidden_dim_post_measurement)
        self.post_measurement_fc = nn.Linear(hidden_dim_post_measurement, 7)

        # Define hidden and cell states
        self.pre_measurement_h = None
        self.pre_measurement_c = None
        self.pre_measurement_hidden_dim = hidden_dim_pre_measurement
        self.post_measurement_h = None
        self.post_measurement_c = None
        self.post_measurement_hidden_dim = hidden_dim_post_measurement

        # Define model outputs
        self.pre_out_vec = None
        self.post_out_vec = None

    def forward(self, img, self_measurement):
        """
        Forward pass for this model

        Args:
            img (torch.Tensor): tensor representing batch of images of shape (N, H, W, C)
            self_measurement (torch.Tensor): tensor representing batch of measurements of active robot's eef state
                of shape (N, 7)

        Returns:
            pre_out (torch.Tensor): output from pre-measurement branch of forward pass, of shape (N, 7)
            post_out (torch.Tensor): output from post-measurement branch of forward pass, of shape (N, 7)
        """
        # TODO: Check to make sure ResNet is in same eval() or train() mode as top level layers
        # First, pass img through ResNet to extract features
        features = self.feature_net(img)

        features_copy = features.clone()

        print(features.shape)

        # Pass features through RNN pre-measurement
        new_pre_measurement_h, new_pre_measurement_c = self.pre_measurement_rnn(
            features, (self.pre_measurement_h[-1], self.pre_measurement_c[-1]))

        print(self.pre_measurement_c[-1])
        print(new_pre_measurement_c)

        # Add new h, c states to corresponding arrays
        self.pre_measurement_h.append(new_pre_measurement_h)
        self.pre_measurement_c.append(new_pre_measurement_c)

        # Run FC layer
        pre_out = self.pre_measurement_fc(new_pre_measurement_h)
        #pre_out = torch.zeros((2,7))


        # Evaluate difference between self measurement and pre_out
        measurement_diff = pre_out - self_measurement

        # Concatenate features with diff measurement
        post_in = torch.cat([features_copy, measurement_diff], dim=1)

        # Pass features through RNN + FC pre-measurement
        new_post_measurement_h, new_post_measurement_c = self.post_measurement_rnn(
            post_in, (self.post_measurement_h[-1], self.post_measurement_c[-1]))
        #post_out = self.post_measurement_fc(new_post_measurement_h.clone())
        post_out = torch.zeros((2,7))

        # Add new h, c states to corresponding arrays
        self.post_measurement_h.append(new_post_measurement_h)
        self.post_measurement_c.append(new_post_measurement_c)

        # Add pre and post out to corresponding arrays
        self.pre_out_vec.append(pre_out.clone())
        self.post_out_vec.append(post_out)

        # Return final output
        return self.pre_out_vec[-1], self.post_out_vec[-1]

    def reset_initial_state(self, batch_size):
        """
        Resets any initial state. Doesn't actually do anything for this class since it doesn't have any temporal
        dependencies

        Args:
            batch_size (int): Batch size currently being run
        """
        self.pre_measurement_h = [torch.zeros((batch_size, self.pre_measurement_hidden_dim), requires_grad=False)]
        self.pre_measurement_c = [torch.zeros((batch_size, self.pre_measurement_hidden_dim), requires_grad=False)]
        self.post_measurement_h = [torch.zeros((batch_size, self.post_measurement_hidden_dim), requires_grad=False)]
        self.post_measurement_c = [torch.zeros((batch_size, self.post_measurement_hidden_dim), requires_grad=False)]

        self.pre_out_vec = []
        self.post_out_vec = []
