from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
from torchvision import models
import matplotlib.pyplot as plt


def visualize_layer(model, layer, img):
    """
    Visualizes the output from a specific layer from a given model

    Args:
        model (nn.Module): NN Module to pass input through
        layer (int): Layer to visualize (from resnet)
            0 results in conv1 layer
            9 results in bn1 layer
        img (torch.Tensor): Image to process through nn.Module. Should be of form (C, H, W)
    """
    # Define local var to hold layer output
    layer_out = None

    # Define hook to run

    def forward_hook(module, input_, output):
        nonlocal layer_out
        layer_out = output.squeeze(dim=0).detach().numpy()

    # Process layer
    if layer == 0:
        layer_name = "conv1"
    elif layer == 9:
        layer_name = "bn1"
    else:
        layer_name = "layer{}".format(layer)

    # Register hook
    getattr(model.feature_net, layer_name).register_forward_hook(forward_hook)

    # Run forward pass
    model.eval()
    img = img.unsqueeze(dim=0)
    model.feature_net(img)

    # Get channel dims
    C, H, W = layer_out.shape

    # Visualize results
    print(layer_out.shape)
    n = int(np.ceil(np.sqrt(C)))

    # Create subplot
    plt.figure()

    plt.imshow(layer_out[92, :, :].squeeze())
    plt.gca().invert_yaxis()

    # Fill in subplot
    #for i in range(C):
    #    plt.subplot(n, n, i+1)
    #    plt.imshow(layer_out[i, :, :].squeeze())
    #    plt.gca().invert_yaxis()

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
    set_parameter_requires_grad(model, feature_extract)

    # Modify final fc layer of resnet
    fc_input_dim = model.fc.in_features
    model.fc = nn.Linear(fc_input_dim, output_dim)

    # Note minimum input size for ResNet
    input_size = 224

    # Return the model and minimum size
    return model, input_size




