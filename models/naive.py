import torch
import torch.nn as nn
import torch.nn.functional as F
from util.model_utils import import_resnet
import math


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


class NaiveObjectStateEstimator(nn.Module):
    """
    A naive estimator for determining an object's pose based on visual observation and active arm's (noisy)
    measurement of its own state. Does not take into account any temporal dependencies (i.e.: this is a one-shot
    regressor)
    """

    def __init__(
            self,
            object_name,
            hidden_dims,
            num_resnet_layers=50,
            latent_dim=50,
            feature_extract=True,
            feature_layer_nums=(9,),
            use_depth=False,
            use_pretrained=True,
            no_proprioception=False,
    ):
        """
        Args:
            object_name (str): name of object to train pose for

            hidden_dims (int or list of ints): Number of FC layers for neural net

            num_resnet_layers (int): Number of layers for imported, pretrained ResNet model.
                Options are 18, 34, 50, 101, 152

            latent_dim (int): Latent space dimension size; this is the output of the ResNet network

            feature_extract (bool): Whether we're feature extracting from ResNet or finetuning

            feature_layer_nums (None or Tuple of int): If not None, determines the additional feature layers to
                concatenate to the main feature output, where each input is the layer number from resnet:
                    Layer to visualize (from resnet)
                        0 results in conv1 layer
                        9 results in bn1 layer

            use_depth (bool): Whether to use depth features or not

            use_pretrained (bool): Whether to use pretrained ResNet model or not

            no_proprioception (bool): If true, creates model that will not leverage proprioceptive measurements
                for state estimation
        """
        # Always run super init first
        super(NaiveObjectStateEstimator, self).__init__()

        # Save relevant args
        self.object_name = object_name
        self.use_proprioception = not no_proprioception

        # Import ResNet as feature model
        self.early_features = None
        self.aux_nets = None
        self.depth_nets = None
        self.aux_latent_dim = 0
        self.use_depth = use_depth
        self.feature_net, _ = import_resnet(
            num_resnet_layers,
            latent_dim,
            feature_extract,
            use_pretrained=use_pretrained
        )

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
                # getattr(self.feature_net, layer_name).register_backward_hook(self.backward_hook)

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
                                *([torch.nn.AvgPool2d(2) for i in range(int(math.log(224 ** 2 / (H * W // 4), 4)))] +
                                  [torch.nn.InstanceNorm2d(1, affine=True), torch.nn.Flatten()])
                            )
                        )
                    )

                    # Add the (flattened) output dimension to the auxiliary variable
                    self.aux_latent_dim += H * W // 4

                # Lastly, reset the early features
                self.early_features = []

            # Lastly, wrap aux and depth nets in nn.ModuleList so it shows up in the appropriate param list
            self.aux_nets = nn.ModuleList(self.aux_nets)
            self.depth_nets = nn.ModuleList(self.depth_nets)

        # Send feature net to DataParallel
        self.feature_net = nn.DataParallel(self.feature_net)

        print("Latent Dim + Aux Dim = {}".format(latent_dim + self.aux_latent_dim))

        # Make sure pre-hidden dims is of type list
        if type(hidden_dims) is int:
            hidden_dims = [hidden_dims]

        # Input dim should be latent_dim + aux_latent_dim + 7 if we're using proprioceptive measurements
        # else just latent_dim + aux_latent_dim
        input_dim = latent_dim + self.aux_latent_dim
        if self.use_proprioception:
            input_dim += 7

        # Define FC layers to run
        fc_dims = [input_dim] + hidden_dims + [7]
        for i, fc_dim in enumerate(fc_dims):
            # We skip the last layer since this is the output
            if i == len(fc_dims) - 1:
                continue
            else:
                setattr(self, "fc{}".format(i), nn.DataParallel(nn.Linear(fc_dim, fc_dims[i+1])))

        # Store number of network layers
        self.n_fc = len(fc_dims) - 1

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
            img (torch.Tensor): tensor representing batch of images of shape (N, C, H, W)
            depth (torch.Tensor): tensor representing batch of depth images of shape (N, 1, H, W)
            self_measurement (torch.Tensor): tensor representing batch of measurements of active robot's eef state
                of shape (N, 7)

        Returns:
            pre_out (torch.Tensor): output from pre-measurement branch of forward pass, of shape (N, 7)
            post_out (torch.Tensor): output from post-measurement branch of forward pass, of shape (N, 7)
        """
        # Get relevant dims
        N = img.shape[0]

        # First, pass img through ResNet to extract features
        features = self.feature_net(img)

        # Combine main features with earlier-layer features if requested
        if self.early_features is not None:
            # Compute auxiliary features
            aux_features = [aux_net(layer_features) for aux_net, layer_features in
                            zip(self.aux_nets, self.early_features)]
            # use depth features if requested
            if self.use_depth:
                # Compute depth features
                depth_features = [depth_net(depth) for depth_net in
                                  self.depth_nets]
                # Combine these features
                aux_features = [aux_feature * depth_feature for aux_feature, depth_feature in
                                zip(aux_features, depth_features)]

            # Concat these synthesized features into main feature array
            features = torch.cat((features, *aux_features), dim=-1)

        # Flatten features
        out = features.view(N, -1)                                 # Output shape (N, latent_dim + aux_dim)

        # Concat these features with measured eef pose if we're using proprioceptive measurements
        if self.use_proprioception:
            out = torch.cat((out, self_measurement), dim=-1)  # Output shape (N, latent_dim + aux_dim + 7)

        # Pass input through FC + Highway layers pre-measurement
        for i in range(self.n_fc):
            # Pass through FC + Activation
            out = F.relu(getattr(self, "fc{}".format(i))(out))

        # Lastly, clear early features variable before returning
        if self.early_features is not None:
            self.early_features = []

        # Return final output
        return out

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
