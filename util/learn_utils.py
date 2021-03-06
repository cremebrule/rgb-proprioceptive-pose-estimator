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

import matplotlib.pyplot as plt

from models.time_sensitive import *
from models.naive import *
from util.data_utils import standardize_quat


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
    train_obj_pose = True if hasattr(model, "object_name") else False

    # Get exact date and time to save model
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")

    # Create variable for storing current time
    since = time.time()

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
            running_pos_err = 0.0
            running_ori_err = 0.0

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
                        pos_err, ori_err = criterion["val_loss"](obj_out, obj)
                    else:
                        x0_out, x1_out = model(img, depth, x0bar)      # Each output is shape (S, N, 7)
                        # Calculate losses
                        loss_x0 = criterion["x0_loss"](x0_out, x0)
                        loss_x1 = criterion["x1_loss"](x1_out, x1)
                        # Sum the losses
                        loss = loss_x0 + loss_x1
                        # Calculate error
                        pos_err, ori_err = criterion["val_loss"](x1_out, x1)

                    # Run backward pass + optimizer step if in training phase
                    if phase == 'train':
                        #print("Running backward step...")
                        loss.backward()
                        optimizer.step()

                # Evaluate statistics as we go along
                running_loss += loss.item()
                running_pos_err += pos_err
                running_ori_err += ori_err

            # Determine overall epoch loss and performance (this is the per-step loss / err averaged over entire epoch)
            epoch_loss = running_loss / (len(dataloader.dataset) * num_episodes)
            epoch_pos_err = running_pos_err / (len(dataloader.dataset) * num_episodes)
            epoch_ori_err = running_ori_err / (len(dataloader.dataset) * num_episodes)

            # Determine current time
            time_elapsed = time.time() - since

            # Add logging stats
            if logging:
                if phase == 'train':
                    writer.add_scalar("Loss/train", epoch_loss, epoch)
                    writer.add_scalar("Err_pos/train", epoch_pos_err, epoch)
                    writer.add_scalar("Err_ori/train", epoch_ori_err, epoch)
                else:  # validation phase
                    writer.add_scalar("Loss/val", epoch_loss, epoch)
                    writer.add_scalar("Err_pos/val", epoch_pos_err, epoch)
                    writer.add_scalar("Err_ori/val", epoch_ori_err, epoch)

            # Print out the stats for this epoch
            if logging:
                print('{} Loss: {:.4f}, PosErr: {:.4f}, OriErr: {:.4f}. Time elapsed = {:.0f}m {:.0f}s'.format(
                    phase, epoch_loss, epoch_pos_err, epoch_ori_err, time_elapsed // 60, time_elapsed % 60))

            # Update val history if this is a val loop
            if phase == 'val':
                # Epoch err is the loss used in this case
                epoch_err = epoch_loss
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
        error_function,
        num_episodes=10,
        model_outputs=None,
        video_writer=None,
        ):
    """
    Performs training for a given model for a specified number of epochs.

    Args:
        model (nn.Module): Model to train
        env (MujocoEnv): robosuite environment to run simulation from
        params (dict): Dict of parameters required for training. Should include entries: "camera_name", "noise_scale"
        motion (str): Type of motion to use. Supported options are "random", "up", and "up_random" currently
                "up_random" only applies to two-armed environments, where the first arm goes up but the second arm
                exhibits random motion
        error_function (nn.Module): Loss / Error function that calculates error metric for the rollout
        num_episodes (int): Number of episodes to run to test
        model_outputs (np.array): Numpy array of shape (N*H, 3), where N=number of episodes and H is steps per episode
            If specified, will automatically send placement indicator to this location at each timestep. If None, then
            a new array will be created and store model outputs, which will be saved to a .npy file
        video_writer (boolimageio.get_writer): If specified, will save frames to this writer

    Returns:
        None
    """
    # Check env type
    is_two_arm = "TwoArm" in str(type(env))

    # Check which model we have so we know what we're evaluating
    eval_obj_pose = True if isinstance(model, TemporallyDependentObjectStateEstimator) or \
        isinstance(model, NaiveObjectStateEstimator) else False

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
    arm_dim = int(len(low) / 2) if is_two_arm else len(low)

    # Setup printing options for numbers
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    # Set model mode to rollout
    model.eval()
    model.rollout = True

    # If model outputs not specified, create list to hold values
    if model_outputs is None:
        model_outputs = []

    # Define global step counter
    global_steps = 0

    # Define total running errors
    total_pos_err = []
    total_ori_err = []

    # Get variable to know when robosuite env is done
    for _ in range(num_episodes):    # Run for specified number of episodes
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

        # Running error
        episode_pos_err = []
        episode_ori_err = []

        # Track number of episode steps
        steps = 0

        # Reset other per-episode vars
        direction = 1

        # Notify user a new episode is starting
        print("\nNEW EPISODE")
        print("*" * 30)

        while not done:
            # Create action based on type specified
            if motion == "up":
                # Move upwards
                action = np.zeros(len(low))
                action[2] = direction * high[2]
                # Also check if we need to switch directions
                if (steps + 1) % 20 == 0:
                    direction = -direction
            else:  # "random" or "up_random"
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
                # If we're doing "up_random", overwrite the first half of the action
                if motion == "up_random":
                    # Move upwards for first arm
                    action[:arm_dim] = np.zeros(arm_dim)
                    action[2] = direction * high[2]
                    # Also check if we need to switch directions
                    if (steps + 1) % 20 == 0:
                        direction = -direction

            # Execute action
            obs, reward, done, _ = env.step(action)

            # Get relevant observations
            # Need to preprocess image first before appending
            img = obs[params["camera_name"] + "_image"]
            # Add this image to video writer if specified
            if video_writer is not None:
                video_writer.append_data(img[::-1])
            #plt.imshow(np.transpose(img, (0,1,2)))
            #plt.show()
            img = transform(img).float()
            x0 = np.concatenate([obs["robot0_eef_pos"], standardize_quat(obs["robot0_eef_quat"])])
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

            # squeeze 1st dim (only if we're not using sequences)
            if not model.requires_sequence:
                img = torch.squeeze(img, dim=0)
                measurement_self = torch.squeeze(measurement_self, dim=0)
                true_self = torch.squeeze(true_self, dim=0)

            # Optionally add in depth observations
            if use_depth:
                depth = obs[params["camera_name"] + "_depth"]
                depth = depth_transform(depth).float().unsqueeze(dim=0).unsqueeze(dim=0)
                # squeeze 1st dim (only if we're not using sequences)
                if not model.requires_sequence:
                    depth = torch.squeeze(depth, dim=0)
            else:
                depth = None

            # Now add model-specific observations
            if eval_obj_pose:
                true_obj = np.concatenate([obs[model.object_name + "_pos"], standardize_quat(obs[model.object_name + "_quat"])])
                true_obj = torch.tensor(true_obj, dtype=torch.float, requires_grad=False).unsqueeze(
                    dim=0).unsqueeze(dim=0)

                # squeeze 1st dim (only if we're not using sequences)
                if not model.requires_sequence:
                    true_obj = torch.squeeze(true_obj, dim=0)

                # Now run forward pass to get estimates
                obj_out = model(img, depth, measurement_self)

                obj_pose = obj_out.squeeze().detach().numpy()
                obj_pose_true = true_obj.squeeze().detach().numpy()

                # Normalize model output quat
                obj_pose[3:] /= np.sqrt(np.sum(np.power(obj_pose[3:], 2)))

                # Calculate error
                pos_err, ori_err = error_function(obj_out, true_obj)

                print("{}: || POS: Est: {}, True: {}, Err: {:.3f} || ORI: Est: {}, True: {}, Err: {:.3f}".format(
                    model.object_name, obj_pose[:3], obj_pose_true[:3], pos_err,
                    obj_pose[3:], obj_pose_true[3:], ori_err)
                )

                # If model outputs is a list, add the model output obj_position to this
                if type(model_outputs) is list:
                    model_outputs.append(obj_pose[:3])
                else:
                    # This is a loaded model outputs; move obj_position to this location
                    # Set the indicator object to this xyz location to visualize
                    env.move_indicator(model_outputs[global_steps])

            else:
                true_other = np.concatenate([obs["robot1_eef_pos"], standardize_quat(obs["robot1_eef_quat"])])
                true_other = torch.tensor(true_other, dtype=torch.float, requires_grad=False).unsqueeze(
                    dim=0).unsqueeze(dim=0)

                # squeeze 1st dim (only if we're not using sequences)
                if not model.requires_sequence:
                    true_other = torch.squeeze(true_other, dim=0)

                # Now run forward pass to get estimates
                x0_out, x1_out = model(img, depth, measurement_self)

                x0_pose = x0_out.squeeze().detach().numpy()
                x0_pose_true = true_self.squeeze().detach().numpy()
                x1_pose = x1_out.squeeze().detach().numpy()
                x1_pose_true = true_other.squeeze().detach().numpy()

                # Normalize model output quat
                x0_pose[3:] /= np.sqrt(np.sum(np.power(x0_pose[3:], 2)))
                x1_pose[3:] /= np.sqrt(np.sum(np.power(x1_pose[3:], 2)))

                # Calculate error
                pos_err, ori_err = error_function(x1_pose, x1_pose_true)

                print("OTHER: || POS: Est: {}, True: {}, Err: {:.3f} || ORI: Est: {}, True: {}, Err: {:.3f}".format(
                    x1_pose[:3], x1_pose_true[:3], pos_err,
                    x1_pose[3:], x1_pose_true[3:], ori_err)
                )

                if type(model_outputs) is list:
                    model_outputs.append(x1_pose[:3])
                else:
                    # This is a loaded model outputs; move obj_position to this location
                    # Set the indicator object to this xyz location to visualize
                    env.move_indicator(model_outputs[global_steps])

            # Add error to running error
            episode_pos_err.append(pos_err)
            episode_ori_err.append(ori_err)

            # Increment episode step and global step
            steps += 1
            global_steps += 1

            # Lastly, render
            #env.render()
            #env.render()

        # Close all plots
        plt.close('all')

        # Save model outputs if we created one this time
        if type(model_outputs) is list:
            with open('model_outputs.npy', 'wb') as f:
                np.save(f, np.array(model_outputs))

        # At the end, print the episode error
        print("EPISODE COMPLETED -- Total Pos/Ori err: {:.3f} m / {:.3f} rad, Per-Step Err: {:.3f} m / {:.3f} rad"
              .format(np.sum(episode_pos_err), np.sum(episode_ori_err), np.average(episode_pos_err), np.average(episode_ori_err)))

        # We also add this episode's errors to the total error vectors
        total_pos_err += episode_pos_err
        total_ori_err += episode_ori_err

    # Finally, we print out the total evaluation metrics
    print()
    print("*" * 90)
    print("EVALUATION COMPLETED -- Per-Step Pos Mean/Std Err: {:.5f} / {:.5f} m || Ori Mean/Std Err: {:.5f} / {:.5f} rad"
          .format(np.average(total_pos_err), np.std(total_pos_err), np.average(total_ori_err), np.std(total_ori_err)))
    print("*" * 90)
