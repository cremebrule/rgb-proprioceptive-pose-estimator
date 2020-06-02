import torch
import torch.nn as nn
import torch.nn.functional as F
from util.model_utils import import_resnet
import math


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
            feature_extract=True,
            feature_layer_nums=(9,),
            use_depth=False,
            device='cpu'
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

            feature_extract (bool): Whether we're feature extracting from ResNet or finetuning

            feature_layer_nums (None or Tuple of int): If not None, determines the additional feature layers to
                concatenate to the main feature output, where each input is the layer number from resnet:
                    Layer to visualize (from resnet)
                        0 results in conv1 layer
                        9 results in bn1 layer

            use_depth (bool): Whether to use depth features or not

            device (str): Device to send all sub modules to
        """
        # Always run super init first
        super(TemporallyDependentStateEstimator, self).__init__()

        # Import ResNet as feature model
        self.early_features = None
        self.aux_nets = None
        self.depth_nets = None
        self.use_depth = use_depth
        self.aux_latent_dim = 0
        self.feature_net, _ = import_resnet(num_resnet_layers, latent_dim, feature_extract)
        # Add additional feature layer outputs if requested
        if feature_layer_nums is not None:
            self.early_features = []
            self.aux_nets = []
            self.depth_nets = []
            # Loop over each layer and add a hook to grab the output from this
            for layer in feature_layer_nums:
                # Process layer
                if layer == 0:
                    layer_name = "conv1"
                elif layer == 9:
                    layer_name = "bn1"
                else:
                    layer_name = "layer{}".format(layer)

                # Register hook
                getattr(self.feature_net, layer_name).register_forward_hook(self.forward_hook)
                #getattr(self.feature_net, layer_name).register_backward_hook(self.backward_hook)

            # Run a dummy forward pass to get the relevant dimensions from each layer
            with torch.no_grad():
                inp = torch.zeros(1, 3, 224, 224)
                self.feature_net(inp)
                # Loop over each cached early feature output
                for feature in self.early_features:
                    # Get relevant dimensions
                    _, C, H, W = feature.shape
                    # Create the auxiliary layer for this layer output
                    self.aux_nets.append(
                        torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=C, out_channels=1, kernel_size=1),
                            torch.nn.MaxPool2d(2),
                            torch.nn.Flatten()
                        )
                    )
                    # Define depth net for this layer
                    self.depth_nets.append(
                        torch.nn.Sequential(
                            *([torch.nn.AvgPool2d(2) for i in range(int(math.log(224**2 / (H*W // 4), 4)))] +
                            [torch.nn.InstanceNorm2d(1, affine=True), torch.nn.Flatten()])
                        )
                    )

                    # Add the (flattened) output dimension to the auxiliary variable
                    self.aux_latent_dim += H*W // 4

                # Lastly, reset the early features
                self.early_features = []

        print("Latent Dim + Aux Dim = {}".format(latent_dim + self.aux_latent_dim))

        # Define LSTM nets
        self.pre_measurement_rnn = nn.LSTM(input_size=latent_dim + self.aux_latent_dim,
                                           hidden_size=hidden_dim_pre_measurement)
        self.pre_measurement_fc = nn.Linear(hidden_dim_pre_measurement, 7)
        self.post_measurement_rnn = nn.LSTM(input_size=latent_dim + self.aux_latent_dim + 7,
                                            hidden_size=hidden_dim_post_measurement)
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

    def forward_hook(self, module, input_, output):
        self.early_features.append(output)

    def backward_hook(self, module, grad_input, grad_output):
        print('Inside ' + self.__class__.__name__ + ' backward')
        print('Inside class:' + self.__class__.__name__)
        print('')
        print('grad_input: ', type(grad_input))
        print('grad_input[0]: ', type(grad_input[0]))
        print('grad_output: ', type(grad_output))
        print('grad_output[0]: ', type(grad_output[0]))
        print('')
        print('grad_input size:', grad_input[0].size())
        print('grad_output size:', grad_output[0].size())
        print('grad_input norm:', grad_input[0].norm())

    def forward(self, img, depth, self_measurement):
        """
        Forward pass for this model

        Args:
            img (torch.Tensor): tensor representing batch of sequences of images of shape (S, N, C, H, W)
            depth (torch.Tensor): tensor representing batch of depth images of shape (S, N, 1, H, W)
            self_measurement (torch.Tensor): tensor representing batch of sequence of measurements of active robot's eef state
                of shape (S, N, 7)

        Returns:
            pre_out (torch.Tensor): output from pre-measurement branch of forward pass, of shape (S, N, 7)
            post_out (torch.Tensor): output from post-measurement branch of forward pass, of shape (S, N, 7)
        """
        # TODO: Check to make sure ResNet is in same eval() or train() mode as top level layers
        # First, reshape imgs before passing through ResNet
        S, N, C, H, W = img.shape
        img = img.view(-1, C, H, W)

        # Pass img through ResNet to extract features
        features = self.feature_net(img)

        # Combine main features with earlier-layer features if requested
        if self.early_features is not None:
            # Compute auxiliary features
            aux_features = [aux_net(layer_features) for aux_net, layer_features in
                            zip(self.aux_nets, self.early_features)]
            if self.use_depth:
                # Reshape depth
                depth = depth.view(-1, 1, H, W)
                # Compute depth features
                depth_features = [depth_net(depth) for depth_net in
                                  self.depth_nets]
                # Combine these features if requested
                aux_features = [aux_feature * depth_feature for aux_feature, depth_feature in
                                  zip(aux_features, depth_features)]
            # Concat these synthesized features into main feature array
            features = torch.cat((features, *aux_features), dim=-1)

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

        # Lastly, clear early features variable before returning
        if self.early_features is not None:
            self.early_features = []

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


class TemporallyDependentObjectStateEstimator(nn.Module):
    """
    A less-naive estimator for determining an object's pose based on visual observation from a robot's "eye-in-hand"
    camera and arm's (noisy) measurement of its own state. Takes into account temporal dependencies via LSTM module
    """

    def __init__(
            self,
            object_name,
            hidden_dim,
            num_resnet_layers=50,
            latent_dim=50,
            sequence_length=10,
            dropout_prob=0.10,
            feature_extract=True,
            feature_layer_nums=(9,),
            use_depth=False,
            device='cpu'
    ):
        """
        Args:
            object_name (str): name of object to train pose for

            hidden_dim (int): size of hidden state for LSTM

            num_resnet_layers (int): Number of layers for imported, pretrained ResNet model.
                Options are 18, 34, 50, 101, 152

            latent_dim (int): Latent space dimension size; this is the output of the ResNet network

            sequence_length (int): Size of sequences to be input into LSTM

            dropout_prob (float): Dropout probability for LSTM layers (TODO: Currently does nothing)

            feature_extract (bool): Whether we're feature extracting from ResNet or finetuning

            feature_layer_nums (None or Tuple of int): If not None, determines the additional feature layers to
                concatenate to the main feature output, where each input is the layer number from resnet:
                    Layer to visualize (from resnet)
                        0 results in conv1 layer
                        9 results in bn1 layer

            use_depth (bool): Whether to use depth features or not

            device (str): Device to send all sub modules to
        """
        # Always run super init first
        super(TemporallyDependentObjectStateEstimator, self).__init__()

        # Save object name
        self.object_name = object_name

        # Import ResNet as feature model
        self.early_features = None
        self.aux_nets = None
        self.depth_nets = None
        self.aux_latent_dim = 0
        self.use_depth = use_depth
        self.feature_net, _ = import_resnet(num_resnet_layers, latent_dim, feature_extract)
        # Add additional feature layer outputs if requested
        if feature_layer_nums is not None:
            self.early_features = []
            self.aux_nets = []
            self.depth_nets = []
            # Loop over each layer and add a hook to grab the output from this
            for layer in feature_layer_nums:
                # Process layer
                if layer == 0:
                    layer_name = "conv1"
                elif layer == 9:
                    layer_name = "bn1"
                else:
                    layer_name = "layer{}".format(layer)

                # Register hook
                getattr(self.feature_net, layer_name).register_forward_hook(self.forward_hook)
                #getattr(self.feature_net, layer_name).register_backward_hook(self.backward_hook)

            # Run a dummy forward pass to get the relevant dimensions from each layer
            with torch.no_grad():
                inp = torch.zeros(1, 3, 224, 224)
                self.feature_net(inp)
                # Loop over each cached early feature output
                for feature in self.early_features:
                    # Get relevant dimensions
                    _, C, H, W = feature.shape
                    # Create the auxiliary layer for this layer output
                    self.aux_nets.append(
                        nn.DataParallel(
                            torch.nn.Sequential(
                                torch.nn.Conv2d(in_channels=C, out_channels=1, kernel_size=1),
                                torch.nn.MaxPool2d(2),
                                torch.nn.Flatten()
                            )
                        )
                    )
                    # Define depth net for this layer
                    self.depth_nets.append(
                        nn.DataParallel(
                            torch.nn.Sequential(
                                *([torch.nn.AvgPool2d(2) for i in range(int(math.log(224**2 / (H*W // 4), 4)))] +
                                [torch.nn.InstanceNorm2d(1, affine=True), torch.nn.Flatten()])
                            )
                        )
                    )

                    # Add the (flattened) output dimension to the auxiliary variable
                    self.aux_latent_dim += H*W // 4

                # Lastly, reset the early features
                self.early_features = []

        # Send feature net to DataParallel
        self.feature_net = nn.DataParallel(self.feature_net)

        print("Latent Dim + Aux Dim = {}".format(latent_dim + self.aux_latent_dim))

        # Define LSTM nets
        self.rnn = nn.DataParallel(nn.LSTM(input_size=latent_dim + self.aux_latent_dim + 7,
                           hidden_size=hidden_dim))
        self.fc = nn.DataParallel(torch.nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim // 4)),
            nn.Linear(int(hidden_dim // 4), 7)
        ))
        self.sequence_length = sequence_length

        # Define hidden and cell states
        self.rnn_h = None
        self.rnn_c = None
        self.hidden_dim = hidden_dim

        # Define model outputs
        self.out_vec = None

        # Set rollout to false by default
        self.rollout = False

    def forward_hook(self, module, input_, output):
        self.early_features.append(output)

    def backward_hook(self, module, grad_input, grad_output):
        print('Inside ' + self.__class__.__name__ + ' backward')
        print('Inside class:' + self.__class__.__name__)
        print('')
        print('grad_input: ', type(grad_input))
        print('grad_input[0]: ', type(grad_input[0]))
        print('grad_output: ', type(grad_output))
        print('grad_output[0]: ', type(grad_output[0]))
        print('')
        print('grad_input size:', grad_input[0].size())
        print('grad_output size:', grad_output[0].size())
        print('grad_input norm:', grad_input[0].norm())

    def forward(self, img, depth, self_measurement):
        """
        Forward pass for this model

        Args:
            img (torch.Tensor): tensor representing batch of sequences of images of shape (S, N, C, H, W)
            depth (torch.Tensor): tensor representing batch of depth images of shape (S, N, 1, H, W)
            self_measurement (torch.Tensor): tensor representing batch of sequence of measurements of active robot's eef state
                of shape (S, N, 7)

        Returns:
            pre_out (torch.Tensor): output from pre-measurement branch of forward pass, of shape (S, N, 7)
            post_out (torch.Tensor): output from post-measurement branch of forward pass, of shape (S, N, 7)
        """
        # TODO: Check to make sure ResNet is in same eval() or train() mode as top level layers
        # First, reshape imgs before passing through ResNet
        S, N, C, H, W = img.shape
        img = img.view(-1, C, H, W)

        # Pass img through ResNet to extract features
        features = self.feature_net(img)

        # Combine main features with earlier-layer features if requested
        if self.early_features is not None:
            # Compute auxiliary features
            aux_features = [aux_net(layer_features) for aux_net, layer_features in
                            zip(self.aux_nets, self.early_features)]
            # use depth features if requested
            if self.use_depth:
                # Reshape depth
                depth = depth.view(-1, 1, H, W)
                # Compute depth features
                depth_features = [depth_net(depth) for depth_net in
                                  self.depth_nets]
                # Combine these features
                aux_features = [aux_feature * depth_feature for aux_feature, depth_feature in
                                  zip(aux_features, depth_features)]

            # Concat these synthesized features into main feature array
            features = torch.cat((features, *aux_features), dim=-1)

        # Reshape features
        features = features.view(S, N, -1)                              # Output shape (S, N, latent_dim + aux_dim)

        # Concat these features with measured eef pose
        features = torch.cat((features, self_measurement), dim=-1)      # Output shape (S, N, latent_dim + aux_dim + 7)

        # Pass features through RNN
        if not self.rollout:
            rnn_h, _ = self.rnn(features)  # Output shape (S, N, pre_hidden_dim)
        else:
            rnn_h, (new_h, new_c) = self.rnn(features, (
                self.rnn_h, self.rnn_c))
            self.rnn_h = new_h
            self.rnn_c = new_c

        # Run FC layer
        out = self.fc(rnn_h)            # Output shape (S, N, 7)

        # Lastly, clear early features variable before returning
        if self.early_features is not None:
            self.early_features = []

        # Return final output
        return out

    def reset_initial_state(self, batch_size):
        """
        Resets any initial state

        Args:
            batch_size (int): Batch size currently being run
        """
        self.rnn_h = torch.zeros((1, batch_size, self.hidden_dim), requires_grad=True)
        self.rnn_c = torch.zeros((1, batch_size, self.hidden_dim), requires_grad=True)

        self.out_vec = []

    @property
    def requires_sequence(self):
        return True
