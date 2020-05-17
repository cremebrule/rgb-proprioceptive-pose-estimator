from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from datetime import datetime
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
        save_path='default',
        save_model=True,
        logging=True,
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
        save_model (bool): Whether to save model periodically or not
        logging (bool): Whether we are logging our results via Tensorboard SummaryWriter or not

    Returns:
        model (nn.Module): Model with best results
        best_err (float): Best val loss from training
    """
    # Get exact date and time to save model
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")

    # Create variable for storing current time
    since = time.time()

    # Store validation history
    val_history = []

    # Store best model and its performance
    best_model = copy.deepcopy(model.state_dict())
    best_err = np.inf       # lower is better

    # Logging if requested
    writer = SummaryWriter() if logging else None

    # Load data into dataloader
    batch_size = model.sequence_length if model.requires_sequence else 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Make sure to send model to appropriate device
    model.to(device)

    # Print filepath we're saving model to
    print("\nFile name saved:\n{}_{}_{}hzn_{}ep_{}.pth\n".format(
                            type(model).__name__,
                            type(dataset.env).__name__,
                            dataset.env.horizon,
                            num_epochs*num_train_episodes_per_epoch,
                            dt_string))

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

            # Reset model
            model.reset_initial_state(num_episodes)

            # Setup arrays to hold model outputs over the episode
            x0_out_vec = []
            x1_out_vec = []

            # Now iterate over data
            print("Running {}...".format(phase))
            for img, x0bar, x0, x1 in dataloader:
                # Pass all inputs to specified device (cpu or gpu)
                img = img.to(device)
                x0bar = x0bar.to(device)
                x0 = x0.to(device)
                x1 = x1.to(device)

                # squeeze 1st dim (only if we're not using sequences)
                if not model.requires_sequence:
                    img = torch.squeeze(img, dim=0)
                    x0bar = torch.squeeze(x0bar, dim=0)
                    x0 = torch.squeeze(x0, dim=0)
                    x1 = torch.squeeze(x1, dim=0)

                # Zero out the parameter gradients
                optimizer.zero_grad()

                #torch.autograd.set_detect_anomaly(True)

                # Run the forward pass, and only track gradients if we're in the training mode
                with torch.set_grad_enabled(phase == 'train'):
                    # Calculate outputs
                    x0_out, x1_out = model(img, x0bar)      # Each output is shape (S, N, 7)

                    # Calculate losses
                    loss_x0 = criterion["x0_loss"](x0_out, x0)
                    loss_x1 = criterion["x1_loss"](x1_out, x1)

                    # Sum the losses
                    loss = loss_x0 + loss_x1

                    # Run backward pass + optimizer step if in training phase
                    if phase == 'train':
                        #print("Running backward step...")
                        loss.backward()
                        optimizer.step()

                # Evaluate statistics as we go along
                running_loss += loss.item()
                running_err += loss_x1.item()

            # Determine overall epoch loss and performance (this is the per-step loss / err averaged over entire epoch)
            epoch_loss = running_loss / (len(dataloader.dataset) * num_episodes)
            epoch_err = running_err / (len(dataloader.dataset) * num_episodes)

            # Determine current time
            time_elapsed = time.time() - since

            # Add logging stats
            if logging:
                if phase == 'train':
                    writer.add_scalar("Loss/train", epoch_loss, epoch)
                    writer.add_scalar("Err/train", epoch_err, epoch)
                else:  # validation phase
                    writer.add_scalar("Loss/val", epoch_loss, epoch)
                    writer.add_scalar("Err/val", epoch_err, epoch)

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

                    # Make sure we actually want to save model first
                    if save_model:

                        # Save it to specified path
                        if save_path == 'default':
                            fdir = os.path.dirname(os.path.abspath(__file__))
                            save_path = os.path.join(fdir, '../log/runs/{}_{}_{}hzn_{}ep_{}.pth'.format(
                                type(model).__name__,
                                type(dataset.env).__name__,
                                dataset.env.horizon,
                                num_epochs*num_train_episodes_per_epoch,
                                dt_string),
                                                     )

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
    return model, best_err


def rollout(
        model,
        env,
        params,
        ):
    """
    Performs training for a given model for a specified number of epochs.

    Args:
        model (nn.Module): Model to train
        env (MujocoEnv): robosuite environment to run simulation from
        params (dict): Dict of parameters required for training. Should include entries: "camera_name", "noise_scale"

    Returns:
        None
    """

    # Create image pre-processor
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Get action limits
    low, high = env.action_spec

    # Setup printing options for numbers
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    # Set model mode to rollout
    model.eval()
    model.rollout = True

    # Get variable to know when robosuite env is done
    while(True):    # Run indefinitely
        # Reset env
        env.reset()

        # Reset model internals
        model.reset_initial_state(batch_size=1)

        # Variable to know when episode is done
        done = False

        while not done:
            # Take a random step in the environment
            action = np.random.uniform(low, high)
            obs, reward, done, _ = env.step(action)

            # Get relevant observations
            # Need to preprocess image first before appending
            img = obs[params["camera_name"] + "_image"]
            img = transform(img).float()
            x0 = np.concatenate([obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
            x0bar = x0 + np.random.multivariate_normal(mean=np.zeros(7), cov=np.eye(7) * params["noise_scale"])
            # Renormalize the orientation part
            mag = np.linalg.norm(x0bar[3:])
            x0bar[3:] /= mag
            measurement_self = x0bar
            true_self = x0
            true_other = np.concatenate([obs["robot1_eef_pos"], obs["robot1_eef_quat"]])

            # Convert inputs to tensors
            img = img.unsqueeze(dim=0).unsqueeze(dim=0)
            measurement_self = torch.tensor(measurement_self, dtype=torch.float, requires_grad=False).unsqueeze(dim=0).unsqueeze(dim=0)
            true_self = torch.tensor(true_self, dtype=torch.float, requires_grad=False).unsqueeze(dim=0).unsqueeze(dim=0)
            true_other = torch.tensor(true_other, dtype=torch.float, requires_grad=False).unsqueeze(dim=0).unsqueeze(dim=0)

            # Now run forward pass to get estimates
            x0_out, x1_out = model(img, measurement_self)

            x0_pos = x0_out.squeeze().detach().numpy()
            x0_pos_true = true_self.squeeze().detach().numpy()
            x1_pos = x1_out.squeeze().detach().numpy()[:3]
            x1_pos_true = true_other.squeeze().detach().numpy()[:3]

            print("SELF: Predicted pos: {}, True pos: {}".format(x0_pos, x0_pos_true))
            #print("OTHER: Predicted pos: {}, True pos: {}".format(x1_pos, x1_pos_true))


            # Set the indicator object to this xyz location to visualize
            env.move_indicator(x1_pos)

            # Lastly, render
            env.render()


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




