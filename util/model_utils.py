from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
from torchvision import models
import matplotlib.pyplot as plt


def visualize_layer(model, layer, img, depth=None):
    """
    Visualizes the output from a specific layer from a given model

    Args:
        model (nn.Module): NN Module to pass input through
        layer (str): Layer to visualize
            String expected to be of form 'ans', where:
                t: type of model feature to visualize, {f, d, a}
                n: feature layer number to visualize {0,...,N}
                s: whether to visualize [s]ingle output of [m]ultiple outputs {s, m}
            if layer begins with 'f', this is assumed to requested a "feature" layer from resnet
                f0 results in conv1 layer
                f1-5 results in layer{N} layer in resnet
                f9 results in bn1 layer
            if layer begins with 'd', this is assumed to be a depth feature
                d0-N results in model.depth_nets[n]'s output being visualized
            if layer begins with 'a', this is assumed to be an auxiliary feature
                a0-N results in model.aux_nets[n]'s output being visualized

        img (torch.Tensor): Image to process through nn.Module. Should be of form (C, H, W)
        depth (torch.Tensor): (Optional) depth tensor if model uses depth features
    """
    # Define local var to hold layer output
    layer_out = None

    # Define hook to run

    def forward_hook(module, input_, output):
        nonlocal layer_out
        layer_out = output.squeeze(dim=0).detach().numpy()

    # Process layer to visualize
    if layer[0] == 'f':
        # This is ResNet
        net = model.feature_net
        if layer[1] == '0':
            layer_name = "conv1"
        elif layer[1] == '9':
            layer_name = "bn1"
        else:
            layer_name = "layer" + layer[1]
        vis_layer = getattr(net.module, layer_name)
    elif layer[0] == 'd':
        # This is Depth Features
        vis_layer = model.depth_nets[int(layer[1])].module
    elif layer[0] == 'a':
        # This is Auxiliary Features
        vis_layer = model.aux_nets[int(layer[1])].module
    else:
        raise ValueError("Layer must begin with 'f', 'd', or 'a'! Got: {}".format(layer[0]))

    # Register hook
    vis_layer.register_forward_hook(forward_hook)

    # Run forward pass
    model.eval()
    img = img.unsqueeze(dim=0)
    model(img, depth, torch.zeros((1, 1, 7)))

    # Get relevant dims based on type of output
    if layer[0] == 'f':
        # This is ResNet convolutional outputs, shape will be (C,H,W)
        C, H, W = layer_out.shape
    elif layer[0] in {'d', 'a'}:
        # This is depth or auxiliary outputs, shape will be (H*W)
        HW = layer_out.shape[-1]
        C, H, W = 1, int(np.sqrt(HW)), int(np.sqrt(HW))
        # We also need to reshape the layer output to visualize
        layer_out = layer_out.reshape(C, H, W)
    else:
        # No other options; use dummy values
        C, H, W = 0, 0, 0

    # If this is the depth or aux layer, we need to reshape the height and width accordingly

    # Visualize results
    print(layer_out.shape)
    n = int(np.ceil(np.sqrt(C)))

    # Create subplot
    plt.figure()

    # Check what type of visualization we're using
    if layer[2] == 's':
        # Single feature visualization
        plt.imshow(layer_out[0, :, :].squeeze())
        plt.gca().invert_yaxis()
    else:
        # multiple features being visualized
        # Fill in subplot
        for i in range(C):
            plt.subplot(n, n, i+1)
            plt.imshow(layer_out[i, :, :].squeeze())
            plt.gca().invert_yaxis()

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.show()


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def import_resnet(num_layers, output_dim, feature_extract=True, use_pretrained=True):
    """
    Helper function to load a ResNet model.

    Args:
        num_layers (int): Specific ResNet model to load, based on number of layers. Options are 18, 34, 50, 101, 152
        output_dim (int): Number of outputs that will replace the final fc layer of ResNet
        feature_extract (bool): Whether we're using ResNet to extract features (only re-train final layer) or fine tune
            the entire model
        use_pretrained (bool): Whether we are loading a pretrained version of ResNet

    Returns:
         Imported ResNet model (nn.Module) and minimum input size required (int)
    """
    options = {18, 32, 50, 101, 152}

    # Verify that num_layers is a valid option
    assert num_layers in options, "Invalid layer size specified. Options are: {}".format(options)

    # Import the requested model
    model = getattr(models, "resnet" + str(num_layers))(pretrained=use_pretrained)
    set_parameter_requires_grad(model, (feature_extract and use_pretrained))

    # Modify final fc layer of resnet
    fc_input_dim = model.fc.in_features
    model.fc = nn.Linear(fc_input_dim, output_dim)

    # Note minimum input size for ResNet
    input_size = 224

    # Return the model and minimum size
    return model, input_size




