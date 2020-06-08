import robosuite as suite
from robosuite.models.tasks import UniformRandomSampler
import torch
import torch.nn as nn
from torchvision import transforms
from models.naive import *
from models.time_sensitive import *
from util.data_utils import MultiEpisodeDataset
from util.model_utils import visualize_layer
import argparse
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# List of available models to train
models = {'n', 'no', 'td', 'tdo'}

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="n", help="Which mode to run. Options are 'n', 'no', 'td', or 'tdo'")
parser.add_argument("--model_path", type=str,
                    default="../log/runs/cube_best.pth",
                    help="Where to load saved dict for model")
parser.add_argument("--controller", type=str, default="OSC_POSE", help="Which controller to use in env")
parser.add_argument("--camera_name", type=str, default="frontview", help="Name of camera to render for observations")
parser.add_argument("--horizon", type=int, default=100, help="Horizon per episode run")
parser.add_argument("--sequence_length", type=int, default=10, help="Sequence length for LSTMs")
parser.add_argument("--noise_scale", type=float, default=0.001, help="Noise scale for self measurements")
parser.add_argument("--latent_dim", type=int, default=1024, help="Dimension of output from ResNet")
parser.add_argument("--hidden_dim", nargs="+", type=int, default=[512], help="Hidden dimensions in FC network (naive only), or LSTM net (td/o only)")
parser.add_argument("--env", type=str, default="TwoArmLift", help="Environment to run")
parser.add_argument("--robots", nargs="+", type=str, default=["Panda", "Sawyer"], help="Which robot(s) to use in the env")
parser.add_argument("--use_placement_initializer", action="store_true", help="Whether to use custom placement initializer")
parser.add_argument("--feature_extract", action="store_true", help="Whether ResNet will be set to feature extract mode or not")
parser.add_argument("--no_proprioception", action="store_true", help="If set, will not leverage proprioceptive measurements during rollout")
parser.add_argument("--use_depth", action="store_true", help="Whether to use depth or not")
parser.add_argument("--use_pretrained", action="store_true", help="Whether to usepretrained ResNet or not")
parser.add_argument("--obj_name", type=str, default=None, help="Object name to generate observations of")
args = parser.parse_args()

# Params to define

# robosuite env params
is_two_arm = True if "TwoArm" in args.env else False
camera_name = args.camera_name
horizon = args.horizon

# Define initialization noise (make separate if we're using multi-arm env)
initialization_noise = "default" if args.model in {'tdo', 'no'} else {"magnitude": 0.5, "type": "uniform"}
if is_two_arm:
    initialization_noise = [initialization_noise, {"magnitude": 0.5, "type": "uniform"}]

# Model params
noise_scale = args.noise_scale
num_resnet_layers = 50
sequence_length = args.sequence_length
feature_layer_nums = (9,)

# Define placement initializer for object
rotation_axis = 'y' if args.obj_name == 'hammer' else 'z'
placement_initializer = UniformRandomSampler(
    x_range=[-0.35, 0.35],
    y_range=[-0.35, 0.35],
    ensure_object_boundary_in_range=False,
    rotation=None,
    rotation_axis=rotation_axis,
) if args.use_placement_initializer else None


# Define robosuite model
controller_config = suite.load_controller_config(default_controller=args.controller)
env = suite.make(
    args.env,
    robots=args.robots,
    #has_renderer=True,
    #render_camera=camera_name,
    has_offscreen_renderer=True,
    camera_depths=args.use_depth,
    use_camera_obs=True,
    horizon=horizon,
    camera_names=camera_name,
    controller_configs=controller_config,
    initialization_noise=initialization_noise,
    placement_initializer=placement_initializer
)


if __name__ == '__main__':

    # First print all options
    print("*" * 20)
    print("Running visualization:")
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
    if args.model == 'n':
        model = NaiveEndEffectorStateEstimator(
            hidden_dims_pre_measurement=args.hidden_dim,
            hidden_dims_post_measurement=args.hidden_dim,
            num_resnet_layers=num_resnet_layers,
            latent_dim=args.latent_dim,
            feature_extract=args.feature_extract,
        )
    elif args.model == 'no':
        model = NaiveObjectStateEstimator(
            object_name=args.obj_name,
            hidden_dims=args.hidden_dim,
            num_resnet_layers=num_resnet_layers,
            latent_dim=args.latent_dim,
            feature_extract=args.feature_extract,
            feature_layer_nums=feature_layer_nums,
            use_depth=args.use_depth,
            use_pretrained=args.use_pretrained,
            no_proprioception=args.no_proprioception,
        )
    elif args.model == 'td':
        model = TemporallyDependentStateEstimator(
            hidden_dim_pre_measurement=args.hidden_dim[0],
            hidden_dim_post_measurement=args.hidden_dim[0],
            num_resnet_layers=num_resnet_layers,
            latent_dim=args.latent_dim,
            sequence_length=sequence_length,
            feature_extract=args.feature_extract,
            feature_layer_nums=feature_layer_nums,
            use_depth=args.use_depth,
            use_pretrained=args.use_pretrained,
            device='cpu'
        )
    elif args.model == 'tdo':
        model = TemporallyDependentObjectStateEstimator(
            object_name=args.obj_name,
            hidden_dim=args.hidden_dim[0],
            num_resnet_layers=num_resnet_layers,
            latent_dim=args.latent_dim,
            sequence_length=sequence_length,
            feature_extract=args.feature_extract,
            feature_layer_nums=feature_layer_nums,
            use_depth=args.use_depth,
            use_pretrained=args.use_pretrained,
            no_proprioception=args.no_proprioception,
            device='cpu'
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

    # Create image and depth pre-processor
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

    # Reset model
    model.reset_initial_state(1)

    while(True):
        # Take a random step in the environment
        #action = np.random.uniform(low, high)
        action = np.zeros(len(low))
        action[2] = high[2]

        obs, reward, done, _ = env.step(action)

        # Grab an image
        img = obs[params["camera_name"] + "_image"]

        # Also grab depth image if requested
        depth = None
        if args.use_depth:
            depth = obs[params["camera_name"] + "_depth"]

        # Get from user which layer to visualize
        layer_num = input("Model layer to visualize: ")

        # Now visualize results!
        # Check if we're visualizing the raw input or an actual layer
        if len(layer_num) == 1:
            # This is a raw image, so we'll just plot it directly
            # i --> image, d --> depth
            name = "image" if layer_num[0] == 'i' else "depth"
            image = img if layer_num[0] == 'i' else depth
            print("Visualizing raw input {}...".format(name))
            plt.axis("off")
            plt.imshow(image)
            plt.gca().invert_yaxis()
            plt.show()
        else:
            # Map img and depth to tensors
            img = transform(img).float().unsqueeze(dim=0)
            if depth is not None:
                depth = transform(depth).float().unsqueeze(dim=0)
            print("Visualizing Layer {}...".format(layer_num))
            visualize_layer(model, layer_num, img, depth)
