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
            dropout_prob=0.10,
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

            dropout_prob (float): Dropout probability for LSTM layers (TODO: Currently does nothing)
        """
        # Always run super init first
        super(TemporallyDependentStateEstimator, self).__init__()

        # Import ResNet as feature model
        self.feature_net, _ = import_resnet(num_resnet_layers, latent_dim)

        # Define LSTM nets
        self.pre_measurement_rnn = nn.LSTM(input_size=latent_dim, hidden_size=hidden_dim_pre_measurement)
        self.pre_measurement_fc = nn.Linear(hidden_dim_pre_measurement, 7)
        self.post_measurement_rnn = nn.LSTM(input_size=latent_dim + 7, hidden_size=hidden_dim_post_measurement)
        self.post_measurement_fc = nn.Linear(hidden_dim_post_measurement, 7)
        self.sequence_length = sequence_length

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

        # Set rollout to false by default
        self.rollout = False

    def forward(self, img, self_measurement):
        """
        Forward pass for this model

        Args:
            img (torch.Tensor): tensor representing batch of sequences of images of shape (S, N, H, W, C)
            self_measurement (torch.Tensor): tensor representing batch of sequence of measurements of active robot's eef state
                of shape (S, N, 7)

        Returns:
            pre_out (torch.Tensor): output from pre-measurement branch of forward pass, of shape (S, N, 7)
            post_out (torch.Tensor): output from post-measurement branch of forward pass, of shape (S, N, 7)
        """
        # TODO: Check to make sure ResNet is in same eval() or train() mode as top level layers
        # First, reshape imgs before passing through ResNet
        S, N, H, W, C = img.shape
        img = img.view(-1, H, W, C)

        # Pass img through ResNet to extract features
        features = self.feature_net(img)

        # Reshape features
        features = features.view(S, N, -1)                              # Output shape (S, N, latent_dim)

        # Copy features for use later
        features_copy = features.clone()

        # Pass features through RNN pre-measurement
        if not self.rollout:
            pre_measurement_h, _ = self.pre_measurement_rnn(features)  # Output shape (S, N, pre_hidden_dim)
        else:
            pre_measurement_h, (new_h, new_c) = self.pre_measurement_rnn(features, (
                self.pre_measurement_h, self.pre_measurement_c))
            self.pre_measurement_h = new_h
            self.pre_measurement_c = new_c
        #pre_measurement_h, (pre_h_t, pre_c_t) = self.pre_measurement_rnn(features, (self.pre_measurement_h[-1], self.pre_measurement_c[-1]))      # Output shape (S, N, pre_hidden_dim)

        #self.pre_measurement_h.append(pre_h_t)
        #self.pre_measurement_c.append(pre_c_t)

        # Run FC layer
        pre_out = self.pre_measurement_fc(pre_measurement_h)            # Output shape (S, N, 7)

        # Evaluate difference between self measurement and pre_out
        measurement_diff = pre_out - self_measurement

        # Concatenate features with diff measurement
        post_in = torch.cat([features_copy, measurement_diff], dim=-1)   # Output shape (S, N, latent_dim + 7)


        # Pass features through RNN + FC pre-measurement
        if not self.rollout:
            post_measurement_h, _ = self.post_measurement_rnn(post_in)  # Output shape (S, N, post_hidden_dim)
        else:
            post_measurement_h, (new_h, new_c) = self.post_measurement_rnn(post_in, (
                self.post_measurement_h, self.post_measurement_c))
            self.post_measurement_h = new_h
            self.post_measurement_c = new_c
        #post_measurement_h, (post_h_t, post_c_t) = self.post_measurement_rnn(post_in, (self.post_measurement_h[-1], self.post_measurement_c[-1]))      # Output shape (S, N, post_hidden_dim)

        #self.post_measurement_h.append(post_h_t)
        #self.post_measurement_c.append(post_c_t)

        # Run FC layer
        post_out = self.post_measurement_fc(post_measurement_h)  # Output shape (S, N, 7)

        # Return final output
        return pre_out, post_out

    def reset_initial_state(self, batch_size):
        """
        Resets any initial state. Doesn't actually do anything for this class since it doesn't have any temporal
        dependencies

        Args:
            batch_size (int): Batch size currently being run
        """
        self.pre_measurement_h = torch.zeros((1, batch_size, self.pre_measurement_hidden_dim), requires_grad=False)
        self.pre_measurement_c = torch.zeros((1, batch_size, self.pre_measurement_hidden_dim), requires_grad=False)
        self.post_measurement_h = torch.zeros((1, batch_size, self.post_measurement_hidden_dim), requires_grad=False)
        self.post_measurement_c = torch.zeros((1, batch_size, self.post_measurement_hidden_dim), requires_grad=False)

        self.pre_out_vec = []
        self.post_out_vec = []

    @property
    def requires_sequence(self):
        return True
