import robosuite as suite
import torch
import torch.nn as nn
from models.naive import NaiveEndEffectorStateEstimator
from util.data_utils import MultiEpisodeDataset
from util.model_utils import train

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Script for testing naive model

# Params to define

# robosuite env params
camera_name = "frontview"
horizon = 10

# Model params
noise_scale = 0.01
num_resnet_layers = 50
latent_dim = 256
pre_hidden_dims = [128, 64]
post_hidden_dims = [128, 64]

# Training params
lr = 0.001
n_epochs = 10
n_train_episodes_per_epoch = 2
n_val_episodes_per_epoch = 1


# Define robosuite model
controller_config = suite.load_controller_config(default_controller="OSC_POSE")
env = suite.make(
    "TwoArmLift",
    robots=["Panda", "Sawyer"],
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    horizon=horizon,
    camera_names=camera_name,
    controller_configs=controller_config
)

# Define loss criterion
criterion = {
    "x0_loss": nn.MSELoss(),
    "x1_loss": nn.MSELoss(),
}


if __name__ == '__main__':

    # Create naive model
    print("Loading model...")
    model = NaiveEndEffectorStateEstimator(
        hidden_dims_pre_measurement=pre_hidden_dims,
        hidden_dims_post_measurement=post_hidden_dims,
        num_resnet_layers=num_resnet_layers,
        latent_dim=latent_dim
    )

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    param_list = list(model.parameters())

    # Make sure we're updating the appropriate weights during optimization
    #for param in param_list:
    #    if param.requires_grad:
    #        print(param.shape)

    # Define the dataset
    print("Loading dataset...")
    dataset = MultiEpisodeDataset(env)

    # Define params to pass to training
    params = {
        "camera_name": camera_name,
        "noise_scale": noise_scale
    }

    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    device = "cuda:0" if use_cuda else "cpu"

    # Now train!
    print("Training...")
    best_model, val_history = train(
        model=model,
        dataset=dataset,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=n_epochs,
        num_train_episodes_per_epoch=n_train_episodes_per_epoch,
        num_val_episodes_per_epoch=n_val_episodes_per_epoch,
        params=params,
        device=device,
    )
