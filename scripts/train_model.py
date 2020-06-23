import robosuite as suite
from robosuite.models.tasks import UniformRandomSampler
from models.naive import *
from models.time_sensitive import *
from models.losses import PoseDistanceLoss
from util.data_utils import MultiEpisodeDataset
from util.learn_utils import train
import argparse

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# List of available models to train
models = {'n', 'no', 'td', 'tdo', 'tdo_v2'}

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="n", help="Which mode to run. Options are 'n', 'no', 'td', or 'tdo'")
parser.add_argument("--controller", type=str, default="OSC_POSE", help="Which controller to use in env")
parser.add_argument("--camera_name", type=str, default="frontview", help="Name of camera to render for observations")
parser.add_argument("--horizon", type=int, default=100, help="Horizon per episode run")
parser.add_argument("--sequence_length", type=int, default=10, help="Sequence length for LSTMs")
parser.add_argument("--noise_scale", type=float, default=0.001, help="Noise scale for self measurements")
parser.add_argument("--latent_dim", type=int, default=1024, help="Dimension of output from ResNet")
parser.add_argument("--hidden_dim", nargs="+", type=int, default=[512], help="Hidden dimensions in FC network (naive only), or LSTM net (td/o only)")
parser.add_argument("--proprio_hidden_dim", type=int, default=64, help="Hidden dimensions in proprio LSTM net (tdo_v2 only)")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for Adam optimizer")
parser.add_argument("--n_train_episodes_per_epoch", type=int, default=10, help="Number of training episodes per epoch")
parser.add_argument("--n_val_episodes_per_epoch", type=int, default=2, help="Number of validation episodes per epoch")
parser.add_argument("--env", type=str, default="TwoArmLift", help="Environment to run")
parser.add_argument("--robots", nargs="+", type=str, default=["Panda", "Sawyer"], help="Which robot(s) to use in the env")
parser.add_argument("--use_placement_initializer", action="store_true", help="Whether to use custom placement initializer")
parser.add_argument("--feature_extract", action="store_true", help="Whether ResNet will be set to feature extract mode or not")
parser.add_argument("--no_proprioception", action="store_true", help="If set, will not leverage proprioceptive measurements during training")
parser.add_argument("--use_depth", action="store_true", help="Whether to use depth or not")
parser.add_argument("--use_pretrained", action="store_true", help="Whether to use pretrained ResNet or not")
parser.add_argument("--obj_name", type=str, default=None, help="Object name to generate observations of")
parser.add_argument("--motion", type=str, default="random", help="Type of robot motion to use")
parser.add_argument("--distance_metric", type=str, default="l2", help="Distance metric to use for loss")
parser.add_argument("--loss_mode", type=str, default="pose", help="Type of loss to use. Options are 'position' or 'pose'")
parser.add_argument("--loss_scale_factor", type=float, default=1.0, help="Scaling factor for Pose loss")
parser.add_argument("--alpha", type=float, default=0.5, help="Orientation loss scaling factor relative to position error")
parser.add_argument("--n_epochs", type=int, default=5000, help="Number of epochs")
parser.add_argument("--load_checkpoint", action="store_true", help="Whether to load prior trained model")
parser.add_argument("--checkpoint_model_path", type=str, default="../log/runs/TemporallyDependentObjectStateEstimator_Lift_20hzn_25000ep_02-06-2020_18-49-01.pth",
                    help="Path to checkpoint .pth file to load into model")
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

# Training params
lr = args.lr
n_train_episodes_per_epoch = args.n_train_episodes_per_epoch
n_val_episodes_per_epoch = args.n_val_episodes_per_epoch

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
    has_renderer=False,
    has_offscreen_renderer=True,
    camera_depths=args.use_depth,
    use_camera_obs=True,
    horizon=horizon,
    camera_names=camera_name,
    controller_configs=controller_config,
    initialization_noise=initialization_noise,
    placement_initializer=placement_initializer
)

# Define loss criterion
criterion = {
    "x0_loss": PoseDistanceLoss(distance_metric=args.distance_metric, scale_factor=args.loss_scale_factor, alpha=args.alpha, mode=args.loss_mode),
    "x1_loss": PoseDistanceLoss(distance_metric=args.distance_metric, scale_factor=args.loss_scale_factor, alpha=args.alpha, mode=args.loss_mode),
    "obj_loss": PoseDistanceLoss(distance_metric=args.distance_metric, scale_factor=args.loss_scale_factor, alpha=args.alpha, mode=args.loss_mode),
    "val_loss": PoseDistanceLoss(mode="val"),
}


if __name__ == '__main__':

    # First print all options
    print("*" * 20)
    print("Running experiment:")
    print()
    print("Model: {}".format(args.model))
    print("Pretrained ResNet: {}".format(args.use_pretrained))
    print("Env: {}".format(args.env))
    print("Robots: {}".format(args.robots))
    print("Motion: {}".format(args.motion))
    print("Depth: {}".format(args.use_depth))
    print("Object: {}".format(args.obj_name))
    print("Horizon: {}".format(horizon))
    print("Initialization Noise: {}".format(initialization_noise))
    print("Noise Scale: {}".format(noise_scale))
    print("Number of Epochs: {}".format(args.n_epochs))
    print("Learning Rate: {}".format(lr))
    print("Loss Mode: {}".format(args.loss_mode))
    print("Loss Scaling Factor: {}".format(args.loss_scale_factor))
    print("Alpha: {}".format(args.alpha))
    print("Distance Metric: {}".format(args.distance_metric))
    print("Feature Extraction: {} ".format(args.feature_extract))
    print("Using Proprioception: {}".format(not args.no_proprioception))
    print("Latent Dim: {}".format(args.latent_dim))
    print("Hidden Dim(s): {}".format(args.hidden_dim))
    if args.model == 'tdo_v2':
        print("Proprio Hidden Dim: {}".format(args.proprio_hidden_dim))
    print("Sequence Length: {}".format(sequence_length))
    print("Feature Layer Nums: {}".format(feature_layer_nums))
    if args.model == 'n':
        print("Pre Hidden Dims: {}".format(args.hidden_dim))
        print("Post Hidden Dims: {}".format(args.hidden_dim))
    elif args.model == 'td':
        print("Pre LSTM Hidden Dim: {}".format(args.hidden_dim))
        print("Post LSTM Hidden Dim: {}".format(args.hidden_dim))
    print()
    print("*" * 20)

    # Make sure model is valid
    assert args.model in models, "Error: Invalid model specified. Options are: {}".format(models)

    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    device = "cuda:0" if use_cuda else "cpu"
    print("Using device: {}".format(device))

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
            device=device
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
            device=device
        )
    elif args.model == 'tdo_v2':
        model = TemporallyDependentObjectStateEstimatorV2(
            object_name=args.obj_name,
            img_hidden_dim=args.hidden_dim[0],
            proprio_hidden_dim=args.proprio_hidden_dim,
            num_resnet_layers=num_resnet_layers,
            latent_dim=args.latent_dim,
            sequence_length=sequence_length,
            feature_extract=args.feature_extract,
            feature_layer_nums=feature_layer_nums,
            use_depth=args.use_depth,
            use_pretrained=args.use_pretrained,
            device=device
        )

    else:
        pass

    # Load the saved parameters if requested
    if args.load_checkpoint:
        model.load_state_dict(torch.load(args.checkpoint_model_path, map_location=torch.device(device)))

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    param_list = list(model.parameters())

    # Define the dataset
    print("Loading dataset...")
    dataset = MultiEpisodeDataset(
        env=env,
        use_depth=args.use_depth,
        obj_name=args.obj_name,
        motion=args.motion,
    )

    # Define params to pass to training
    params = {
        "camera_name": camera_name,
        "noise_scale": noise_scale
    }

    # Now train!
    print("Training...")
    best_model, best_val_err = train(
        model=model,
        dataset=dataset,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.n_epochs,
        num_train_episodes_per_epoch=n_train_episodes_per_epoch,
        num_val_episodes_per_epoch=n_val_episodes_per_epoch,
        params=params,
        device=device,
        save_model=True,
    )
