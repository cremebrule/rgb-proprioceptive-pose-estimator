import robosuite as suite
import torch
import torch.nn as nn
from models.naive import NaiveEndEffectorStateEstimator
from models.time_sensitive import TemporallyDependentStateEstimator
from util.data_utils import MultiEpisodeDataset
from util.model_utils import train
import argparse

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# List of available models to train
models = {'naive', 'td'}

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="naive", help="Which mode to run. Options are 'naive' or 'td'")
args = parser.parse_args()

# Params to define

# robosuite env params
camera_name = "frontview"
horizon = 100

# Model params
noise_scale = 0.01
num_resnet_layers = 50
latent_dim = 256
pre_hidden_dims = [128, 64]
post_hidden_dims = [128, 64]
pre_lstm_h_dim = 50
post_lstm_h_dim = 50
sequence_length = 10

# Training params
lr = 0.001
n_epochs = 1000
n_train_episodes_per_epoch = 10
n_val_episodes_per_epoch = 2


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
    "x0_loss": nn.MSELoss(reduction='sum'),
    "x1_loss": nn.MSELoss(reduction='sum'),
}


if __name__ == '__main__':

    # First print all options
    print("*" *  20)
    print("Running experiment:")
    print()
    print("Model: {}".format(args.model))
    print("Horizon: {}".format(horizon))
    print("Noise Scale: {}".format(noise_scale))
    print()
    print("*" *  20)

    # Make sure model is valid
    assert args.model in models, "Error: Invalid model specified. Options are: {}".format(models)

    # Create model
    print("Loading model...")
    model = None
    if args.model == 'naive':
        model = NaiveEndEffectorStateEstimator(
            hidden_dims_pre_measurement=pre_hidden_dims,
            hidden_dims_post_measurement=post_hidden_dims,
            num_resnet_layers=num_resnet_layers,
            latent_dim=latent_dim
        )
    elif args.model == 'td':
        model = TemporallyDependentStateEstimator(
            hidden_dim_pre_measurement=pre_lstm_h_dim,
            hidden_dim_post_measurement=post_lstm_h_dim,
            num_resnet_layers=num_resnet_layers,
            latent_dim=latent_dim,
            sequence_length=sequence_length,
        )
    else:
        pass

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