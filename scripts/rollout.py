import robosuite as suite
from robosuite.models.tasks import UniformRandomSampler
import torch
import torch.nn as nn
from models.naive import NaiveEndEffectorStateEstimator
from models.time_sensitive import *
from util.data_utils import MultiEpisodeDataset
from util.learn_utils import rollout
import argparse

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# List of available models to train
models = {'naive', 'td', 'tdo'}

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="naive", help="Which mode to run. Options are 'naive' or 'td'")
parser.add_argument("--model_path", type=str,
                    default="../log/runs/TemporallyDependentObjectStateEstimator_Lift_100hzn_10000ep_31-05-2020_20-26-58.pth",
                    help="Where to load saved dict for model")
parser.add_argument("--controller", type=str, default="OSC_POSE", help="Which controller to use in env")
parser.add_argument("--camera_name", type=str, default="frontview", help="Name of camera to render for observations")
parser.add_argument("--horizon", type=int, default=100, help="Horizon per episode run")
parser.add_argument("--sequence_length", type=int, default=10, help="Sequence length for LSTMs")
parser.add_argument("--noise_scale", type=float, default=0.001, help="Noise scale for self measurements")
parser.add_argument("--latent_dim", type=int, default=1024, help="Dimension of output from ResNet")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for Adam optimizer")
parser.add_argument("--n_train_episodes_per_epoch", type=int, default=10, help="Number of training episodes per epoch")
parser.add_argument("--n_val_episodes_per_epoch", type=int, default=2, help="Number of validation episodes per epoch")
parser.add_argument("--env", type=str, default="TwoArmLift", help="Environment to run")
parser.add_argument("--robots", nargs="+", type=str, default=["Panda", "Sawyer"], help="Which robot(s) to use in the env")
parser.add_argument("--use_placement_initializer", action="store_true", help="Whether to use custom placement initializer")
parser.add_argument("--feature_extract", action="store_true", help="Whether ResNet will be set to feature extract mode or not")
parser.add_argument("--use_depth", action="store_true", help="Whether to use depth or not")
parser.add_argument("--obj_name", type=str, default=None, help="Object name to generate observations of")
args = parser.parse_args()

# Params to define

# robosuite env params
camera_name = args.camera_name
horizon = args.horizon
initialization_noise = "default" if args.model == 'tdo' else {"magnitude": 0.5, "type": "uniform"}

# Model params
noise_scale = args.noise_scale
num_resnet_layers = 50
feature_extract = False
latent_dim = args.latent_dim
pre_hidden_dims = [512, 128]
post_hidden_dims = [512, 128]
pre_lstm_h_dim = 512
post_lstm_h_dim = 512
sequence_length = args.sequence_length
feature_layer_nums = (9,)

# Model params for tdo
hidden_dim = 512

# Training params
lr = args.lr
n_epochs = 1000
n_train_episodes_per_epoch = args.n_train_episodes_per_epoch
n_val_episodes_per_epoch = args.n_val_episodes_per_epoch

# Define placement initializer for object
placement_initializer = UniformRandomSampler(
    x_range=[-0.375, 0.375],
    y_range=[-0.375, 0.375],
    ensure_object_boundary_in_range=False,
    z_rotation=None,
) if args.use_placement_initializer else None


# Define robosuite model
controller_config = suite.load_controller_config(default_controller=args.controller)
env = suite.make(
    args.env,
    robots=args.robots,
    has_renderer=True,
    has_offscreen_renderer=True,
    camera_depths=args.use_depth,
    use_camera_obs=True,
    horizon=horizon,
    camera_names=camera_name,
    controller_configs=controller_config,
    use_indicator_object=True,
    initialization_noise=initialization_noise,
    placement_initializer=placement_initializer
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
            latent_dim=latent_dim,
            feature_extract=args.feature_extract,
        )
    elif args.model == 'td':
        model = TemporallyDependentStateEstimator(
            hidden_dim_pre_measurement=pre_lstm_h_dim,
            hidden_dim_post_measurement=post_lstm_h_dim,
            num_resnet_layers=num_resnet_layers,
            latent_dim=latent_dim,
            sequence_length=sequence_length,
            feature_extract=args.feature_extract,
            feature_layer_nums=feature_layer_nums,
            use_depth=args.use_depth,
        )
    elif args.model == 'tdo':
        model = TemporallyDependentObjectStateEstimator(
            object_name=args.obj_name,
            hidden_dim=hidden_dim,
            num_resnet_layers=num_resnet_layers,
            latent_dim=latent_dim,
            sequence_length=sequence_length,
            feature_extract=args.feature_extract,
            feature_layer_nums=feature_layer_nums,
            use_depth=args.use_depth,
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
