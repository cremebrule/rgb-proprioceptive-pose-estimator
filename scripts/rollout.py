import robosuite as suite
import torch
import torch.nn as nn
from models.naive import NaiveEndEffectorStateEstimator
from models.time_sensitive import TemporallyDependentStateEstimator
from util.data_utils import MultiEpisodeDataset
from util.model_utils import rollout
import argparse

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# List of available models to train
models = {'naive', 'td'}

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="naive", help="Which mode to run. Options are 'naive' or 'td'")
parser.add_argument("--model_path", type=str,
                    default="../log/runs/TemporallyDependentStateEstimator_TwoArmLift_100hzn_10000ep_18-05-2020_12-21-39.pth",
                    help="Where to load saved dict for model")
args = parser.parse_args()

# Params to define

# robosuite env params
camera_name = "frontview"
horizon = 40
initialization_noise = {"magnitude": 0.5, "type": "uniform"}

# Model params
noise_scale = 0.01
num_resnet_layers = 50
latent_dim = 1024
pre_hidden_dims = [128, 64]
post_hidden_dims = [128, 64]
pre_lstm_h_dim = 512
post_lstm_h_dim = 512
sequence_length = 10


# Define robosuite model
controller_config = suite.load_controller_config(default_controller="OSC_POSE")
env = suite.make(
    "TwoArmLift",
    robots=["Panda", "Sawyer"],
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    horizon=horizon,
    render_camera=camera_name,
    camera_names=camera_name,
    controller_configs=controller_config,
    use_indicator_object=True,
    initialization_noise=initialization_noise,
)


if __name__ == '__main__':

    # First print all options
    print("*" *  20)
    print("Running rollout:")
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

    # Load the saved parameters
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))

    # Define params to pass to training
    params = {
        "camera_name": camera_name,
        "noise_scale": noise_scale
    }

    # Now rollout!
    print("Rollout...")
    rollout(
        model=model,
        env=env,
        params=params,
    )
