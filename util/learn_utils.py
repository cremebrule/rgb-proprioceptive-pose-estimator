from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
import os
import copy

from models.time_sensitive import TemporallyDependentObjectStateEstimator


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
    # Check which model we have so we know what we're training
    train_obj_pose = True if isinstance(model, TemporallyDependentObjectStateEstimator) else False

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
    if device != "cpu":
        model.cuda()

    # Print filepath we're saving model to
    if save_model:
        print("\nFile name saved:\n{}_{}_{}hzn_{}ep_{}.pth\n".format(
                                type(model).__name__,
                                type(dataset.env).__name__,
                                dataset.env.horizon,
                                num_epochs*num_train_episodes_per_epoch,
                                dt_string))

    # Train model over requested number of epochs
    for epoch in range(num_epochs):
        # Notify user
        if logging:
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
            obj_out_vec = []

            # Now iterate over data
            if logging:
                print("Running {}...".format(phase))
            for img, depth, x0bar, x0, x1, obj in dataloader:
                # Pass all inputs to cuda if requested
                if device != "cpu":
                    img = img.cuda()
                    depth = depth.cuda()
                    x0bar = x0bar.cuda()
                    x0 = x0.cuda()
                    if train_obj_pose:
                        obj = obj.cuda()
                    else:
                        x1 = x1.cuda()

                # squeeze 1st dim (only if we're not using sequences)
                if not model.requires_sequence:
                    img = torch.squeeze(img, dim=0)
                    depth = torch.squeeze(depth, dim=0)
                    x0bar = torch.squeeze(x0bar, dim=0)
                    x0 = torch.squeeze(x0, dim=0)
                    if train_obj_pose:
                        obj = torch.squeeze(obj, dim=0)
                    else:
                        x1 = torch.squeeze(x1, dim=0)

                # Zero out the parameter gradients
                optimizer.zero_grad()

                #torch.autograd.set_detect_anomaly(True)

                # Run the forward pass, and only track gradients if we're in the training mode
                with torch.set_grad_enabled(phase == 'train'):
                    # Calculate outputs
                    if train_obj_pose:
                        obj_out = model(img, depth, x0bar)
                        # Calculate loss
                        loss = criterion["obj_loss"](obj_out, obj)
                        # Calculate error
                        err = criterion["val_loss"](obj_out, obj)
                    else:
                        x0_out, x1_out = model(img, depth, x0bar)      # Each output is shape (S, N, 7)
                        # Calculate losses
                        loss_x0 = criterion["x0_loss"](x0_out, x0)
                        loss_x1 = criterion["x1_loss"](x1_out, x1)
                        # Sum the losses
                        loss = loss_x0 + loss_x1
                        # Calculate error
                        err = criterion["val_loss"](x1_out, x1)

                    # Run backward pass + optimizer step if in training phase
                    if phase == 'train':
                        #print("Running backward step...")
                        loss.backward()
                        optimizer.step()

                # Evaluate statistics as we go along
                running_loss += loss.item()
                running_err += err.item()

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
            if logging:
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
    if logging:
        print('-' * 10)

    # Print out resulting stats
    if logging:
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
        motion,
        ):
    """
    Performs training for a given model for a specified number of epochs.

    Args:
        model (nn.Module): Model to train
        env (MujocoEnv): robosuite environment to run simulation from
        params (dict): Dict of parameters required for training. Should include entries: "camera_name", "noise_scale"
        motion (str): Type of motion to use. Supported options are "random" and "up" currently

    Returns:
        None
    """
    # Check which model we have so we know what we're evaluating
    eval_obj_pose = True if isinstance(model, TemporallyDependentObjectStateEstimator) else False

    # Check whether model uses depth or not
    use_depth = model.use_depth

    # Create image pre-processor
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    depth_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
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

        # Auto-variables to reset per episode
        sub_traj_steps = 10
        sub_traj_ct = 0
        action = np.random.uniform(low, high)

        # Variable to know when episode is done
        done = False

        while not done:
            # Create action based on type specified
            if motion == "random":
                # Grab random action for entire action space (only once every few substeps!)
                if sub_traj_ct == sub_traj_steps:
                    # Re-sample action
                    action = np.random.uniform(low, high)
                    # Reset traj counter and re-sample substeps count
                    sub_traj_ct = 0
                    sub_traj_steps = np.random.randint(5, 15)
                else:
                    # increment counter
                    sub_traj_ct += 1
            else:  # type "up"
                # Move upwards
                action = np.zeros(len(low))
                action[2] = high[2]

            # Execute action
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
            # Convert inputs to tensors
            img = img.unsqueeze(dim=0).unsqueeze(dim=0)
            measurement_self = torch.tensor(measurement_self, dtype=torch.float, requires_grad=False).unsqueeze(dim=0).unsqueeze(dim=0)
            true_self = torch.tensor(true_self, dtype=torch.float, requires_grad=False).unsqueeze(dim=0).unsqueeze(dim=0)

            # Optionally add in depth observations
            if use_depth:
                depth = obs[params["camera_name"] + "_depth"]
                depth = depth_transform(depth).float().unsqueeze(dim=0).unsqueeze(dim=0)
            else:
                depth = None

            # Now add model-specific observations
            if eval_obj_pose:
                true_obj = np.concatenate([obs[model.object_name + "_pos"], obs[model.object_name + "_quat"]])
                true_obj = torch.tensor(true_obj, dtype=torch.float, requires_grad=False).unsqueeze(
                    dim=0).unsqueeze(dim=0)

                # Now run forward pass to get estimates
                obj_out = model(img, depth, measurement_self)

                obj_pose = obj_out.squeeze().detach().numpy()
                obj_pose_true = true_obj.squeeze().detach().numpy()
                obj_position = obj_pose[:3]
                obj_position_true = obj_pose_true[:3]

                print("OBJECT: Predicted pos: {}, True pos: {}".format(obj_position, obj_position_true))

                # Set the indicator object to this xyz location to visualize
                # env.move_indicator(x0_pos[:3])
                env.move_indicator(obj_position)

            else:
                true_other = np.concatenate([obs["robot1_eef_pos"], obs["robot1_eef_quat"]])
                true_other = torch.tensor(true_other, dtype=torch.float, requires_grad=False).unsqueeze(
                    dim=0).unsqueeze(dim=0)

                # Now run forward pass to get estimates
                x0_out, x1_out = model(img, depth, measurement_self)

                x0_pos = x0_out.squeeze().detach().numpy()
                x0_pos_true = true_self.squeeze().detach().numpy()
                x1_pos = x1_out.squeeze().detach().numpy()[:3]
                x1_pos_true = true_other.squeeze().detach().numpy()[:3]

                #print("SELF: Predicted pos: {}, True pos: {}".format(x0_pos, x0_pos_true))
                print("OTHER: Predicted pos: {}, True pos: {}".format(x1_pos, x1_pos_true))


                # Set the indicator object to this xyz location to visualize
                #env.move_indicator(x0_pos[:3])
                env.move_indicator(x1_pos)

            # Lastly, render
            #env.render()
            #env.render()
