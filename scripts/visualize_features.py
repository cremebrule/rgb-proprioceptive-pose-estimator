import robosuite as suite
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from models.naive import NaiveEndEffectorStateEstimator
from models.time_sensitive import TemporallyDependentStateEstimator
from util.data_utils import MultiEpisodeDataset
from util.model_utils import visualize_layer
import argparse
import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# List of available models to train
models = {'naive', 'td'}

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="naive", help="Which mode to run. Options are 'naive' or 'td'")
parser.add_argument("--model_path", type=str,
                    default="../log/runs/TemporallyDependentStateEstimator_TwoArmLift_100hzn_10000ep_17-05-2020_00-02-34.pth",
                    help="Where to load saved dict for model")
args = parser.parse_args()

# Params to define

# robosuite env params
camera_name = "frontview"
horizon = 100
initialization_noise = {"magnitude": 0.5, "type": "uniform"}

# Model params
noise_scale = 0.01
num_resnet_layers = 50
latent_dim = 256
pre_hidden_dims = [128, 64]
post_hidden_dims = [128, 64]
pre_lstm_h_dim = 128
post_lstm_h_dim = 128
sequence_length = 10


# Define robosuite model
controller_config = suite.load_controller_config(default_controller="OSC_POSE")
env = suite.make(
    "TwoArmLift",
    robots=["Panda", "Sawyer"],
    has_renderer=False,
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
    print("*" * 20)
    print("Running rollout:")
    print()
    print("Model: {}".format(args.model))
    print("Horizon: {}".format(horizon))
    print("Noise Scale: {}".format(noise_scale))
    print()
    print("*" * 20)

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
    model.load_state_dict(torch.load(args.model_path))

    # Define params to pass to training
    params = {
        "camera_name": camera_name,
        "noise_scale": noise_scale
    }

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

    # Reset model
    model.reset_initial_state(1)


    while(True):
        # Take a random step in the environment
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)

        # Grab an image
        img = obs[params["camera_name"] + "_image"]
        img = transform(img).float()

        # Get from user which layer to visualize
        layer_num = eval(input("ResNet layer number: "))

        # Now visualize results!
        print("Visualizing Layer {}...".format(layer_num))
        visualize_layer(model, layer_num, img)
