import robosuite as suite
from robosuite.models.tasks import UniformRandomSampler
import torch
import torch.nn as nn
from models.naive import *
from models.time_sensitive import *
from models.losses import PoseDistanceLoss
from util.data_utils import MultiEpisodeDataset
from util.learn_utils import rollout
import argparse
import imageio
import numpy as np

from signal import signal, SIGINT
from sys import exit

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# List of available models to train
models = {'n', 'no', 'td', 'tdo'}

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="n", help="Which mode to run. Options are 'n', 'no', 'td', or 'tdo'")
parser.add_argument("--model_path", type=str,
                    default="../log/runs/TemporallyDependentObjectStateEstimator_Lift_20hzn_25000ep_07-06-2020_13-07-19.pth",
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
parser.add_argument("--motion", type=str, default="random", help="Type of robot motion to use")
parser.add_argument("--model_outputs_file", type=str, default=None, help="Path to model outputs to load")
parser.add_argument("--record_video", action="store_true", help="If specified, records video")
args = parser.parse_args()

# Set numpy and torch seed
np.random.seed(3)
torch.manual_seed(3)

# Define callbacks
video_writer = imageio.get_writer("test.mp4", fps=10) if args.record_video else None


def handler(signal_received, frame):
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected. Closing video writer and exiting gracefully')
    video_writer.close()
    exit(0)


# Tell Python to run the handler() function when SIGINT is recieved
signal(SIGINT, handler)

# Load model outputs if defined
if args.model_outputs_file is not None:
    with open(args.model_outputs_file, 'rb') as f:
        model_outputs = np.load(f)
else:
    model_outputs = None


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
    use_indicator_object=args.record_video,
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

    """
    #### DEBUGGING ##
    # Check params
    import numpy as np
    model1_params = list(model.parameters())

    model2 = TemporallyDependentObjectStateEstimator(
        object_name=args.obj_name,
        hidden_dim=args.hidden_dim,
        num_resnet_layers=num_resnet_layers,
        latent_dim=latent_dim,
        sequence_length=sequence_length,
        feature_extract=args.feature_extract,
        feature_layer_nums=feature_layer_nums,
        use_depth=args.use_depth,
    )
    model2.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    model2_params = list(model2.parameters())

    print("Length saved params: {}".format(len(model1_params)))

    for param1, param2 in zip(model1_params, model2_params):
        print("requires grad: {}".format(param1.requires_grad))
        print("norm: {}".format(np.linalg.norm(param1.data.numpy() - param2.data.numpy())))

    print("Checking aux nets...")
    for aux1, aux2 in zip(model.aux_nets, model2.aux_nets):
        for param1, param2 in zip(list(aux1.parameters()), list(aux2.parameters())):
            print("requires grad: {}".format(param1.requires_grad))
            print("norm: {}".format(np.linalg.norm(param1.data.numpy() - param2.data.numpy())))
            #print("norm: {}".format(aux1.weight.data.numpy() - aux2.weight.data.numpy()))

    print("Checking depth nets...")
    for aux1, aux2 in zip(model.depth_nets, model2.depth_nets):
        for param1, param2 in zip(list(aux1.parameters()), list(aux2.parameters())):
            print("requires grad: {}".format(param1.requires_grad))
            print("param1 data: {}".format(param1.data.numpy()))
            print("norm: {}".format(np.linalg.norm(param1.data.numpy() - param2.data.numpy())))

    #### END DEBUGGING ##
    #exit(0)
    """


    # Define params to pass to training
    params = {
        "camera_name": camera_name,
        "noise_scale": noise_scale
    }

    # Define eval error function
    error_function = PoseDistanceLoss(mode="val")

    # Now rollout!
    print("Rollout...")
    rollout(
        model=model,
        env=env,
        params=params,
        motion=args.motion,
        error_function=error_function,
        num_episodes=10,
        model_outputs=model_outputs,
        video_writer=video_writer,
    )
