from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


def train(
        model,
        dataset,
        criterion,
        optimizer,
        num_epochs,
        num_train_episodes_per_epoch,
        num_val_episodes_per_epoch,
        params,
        device,
        save_path='default'
        ):
    """
    Performs training for a given model for a specified number of epochs.

    Args:
        model (nn.Module): Model to train
        dataset (MultiEpisodeDataset): A custom dataset for generating new data online
        criterion (dict): Dict of loss functions to use for training.
            Should include entries: "x0_loss", "x1_loss"
        optimizer (torch.optim.Optimizer): Optimizer to use to automatically update model weights
        num_epochs (int): Number of epochs to run during training
        num_train_episodes_per_epoch (int): Number of episodes to run in parallel per epoch
        num_val_episodes_per_epoch (int): Number of episodes to run in parallel per val run per epoch
        params (dict): Dict of parameters required for training. Should include entries: "camera_name", "noise_scale"
        device (str): Device to send model to for computations. Can be "cpu" or "cuda:X"
        save_path (str): filepath to where best model state dict will be periodically saved

    Returns:
        model (nn.Module): Model with best results
        val_history (list): List of statistics over training run
    """

    # Create variable for storing current time
    since = time.time()

    # Store validation history
    val_history = []

    # Store best model and its performance
    best_model = copy.deepcopy(model.state_dict())
    best_err = np.inf       # lower is better

    # Load data into dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # Train model over requested number of epochs
    for epoch in range(num_epochs):
        # Notify user
        print("\n" + "-" * 10)
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # We will conduct both training and validation phases
        for phase in ['train', 'val']:
            # Set model mode and num episodes based on phase
            if phase == 'train':
                model.train()
                num_episodes = num_train_episodes_per_epoch
            else:
                model.eval()
                num_episodes = num_val_episodes_per_epoch

            # Reset running loss and performance error
            running_loss = 0.0
            running_err = 0.0

            # Grab new data from dataset
            dataloader.dataset.refresh_data(num_episodes, params["camera_name"], params["noise_scale"])

            # Now iterate over data
            for img, x0bar, x0, x1 in dataloader:
                # Pass all inputs to specified device (cpu or gpu) and squeeze 1st dim
                img = torch.squeeze(img.to(device), dim=0)
                x0bar = torch.squeeze(x0bar.to(device), dim=0)
                x0 = torch.squeeze(x0.to(device), dim=0)
                x1 = torch.squeeze(x1.to(device), dim=0)

                # Zero out the parameter gradients
                optimizer.zero_grad()

                # Run the forward pass, and only track gradients if we're in the training mode
                with torch.set_grad_enabled(phase == 'train'):
                    # Calculate outputs
                    x0_out, x1_out = model(img, x0bar)

                    # Calculate losses
                    loss_x0 = criterion["x0_loss"](x0_out, x0)
                    loss_x1 = criterion["x1_loss"](x1_out, x1)

                    # Sum the losses
                    loss = loss_x0 + loss_x1

                    # Run backward pass + optimizer step if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Evaluate statistics as we go along
                running_loss += loss.item() * x0bar.size(0)
                running_err += loss_x1

            # Determine overall epoch loss and performance
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_err = running_err / len(dataloader.dataset)

            # Determine current time
            time_elapsed = time.time() - since

            # Print out the stats for this epoch
            print('{} Loss: {:.4f}, Err: {:.4f}. Time elapsed = {:.0f}m {:.0f}s'.format(
                phase, epoch_loss, epoch_err, time_elapsed // 60, time_elapsed % 60))

            # Update val history if this is a val loop
            if phase == 'val':
                val_history.append(epoch_err)

                # Save this model if it is the best performing one
                if epoch_err < best_err:
                    best_err = epoch_err
                    best_model = copy.deepcopy(model.state_dict())

                    # Save it to specified path
                    if save_path == 'default':
                        fdir = os.path.dirname(os.path.abspath(__file__))
                        save_path = os.path.join(fdir, '../log/runs/{}_{}ep.pth'.format(
                            type(dataloader.dataset.env).__name__, num_epochs*num_train_episodes_per_epoch))

                    # Make sure path exists, if not, create the nested directory to the location
                    directory = os.path.dirname(save_path)
                    try:
                        os.stat(directory)
                    except:
                        os.makedirs(directory)

                    # Lastly, save model dict
                    torch.save(model.state_dict(), save_path)

    # Notify user we're done!!
    print('-' * 10)

    # Print out resulting stats
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Err: {:.4f}'.format(best_err))

    # Load and return the model with the best weights
    model.load_state_dict(best_model)
    return model, val_history


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



