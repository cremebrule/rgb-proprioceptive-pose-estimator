import optuna
import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import robosuite as suite

from models.naive import NaiveEndEffectorStateEstimator
from models.time_sensitive import TemporallyDependentStateEstimator
from models.losses import PoseDistanceLoss
from util.data_utils import MultiEpisodeDataset
from util.model_utils import train

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

models = {'naive', 'td'}

parser = argparse.ArgumentParser(description='RL hyperparam search')
parser.add_argument(
    '--env',
    type=str,
    default='TwoArmLift',
    help='Robosuite env to run test on')
parser.add_argument(
    '--model',
    type=str,
    default='td',
    help='Model to run (see available models above)'
)
parser.add_argument(
    '--robots',
    nargs="+",
    type=str,
    default='Panda',
    help='Robot(s) to run test with')
parser.add_argument(
    '--camera_name',
    type=str,
    default='frontview',
    help='Camera name to grab observations from')
parser.add_argument(
    '--horizon',
    type=int,
    default=100,
    help='max num of timesteps for each simulation')
parser.add_argument(
    '--controller',
    type=str,
    default="OSC_POSE",
    help='controller to use for robot environment. Either name of controller for default config or filepath to custom'
         'controller config')
parser.add_argument(
    '--n_epochs',
    type=int,
    default=10,
    help='Num epochs to run per optimizer trial')
parser.add_argument(
    '--n_episodes_per_epoch',
    type=int,
    default=10,
    help='Num episodes to run per epoch')
parser.add_argument(
    '--n_val_episodes_per_epoch',
    type=int,
    default=2,
    help='Num eval episodes to run for each trial run')
parser.add_argument(
    '--noise_scale',
    type=float,
    default=0.01,
    help='Noise scale for grabbing self measurements')
parser.add_argument(
    '--num_resnet_layers',
    type=int,
    default=50,
    help='ResNet # layers model to use'
)
parser.add_argument(
    '--n_trials',
    type=int,
    default=100,
    help='Number of trials to run'
)
args = parser.parse_args()

# Define robosuite model
controller_config = suite.load_controller_config(default_controller=args.controller)
env = suite.make(
    args.env,
    robots=args.robots,
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    horizon=args.horizon,
    camera_names=args.camera_name,
    controller_configs=controller_config
)

# Define dataset
dataset = MultiEpisodeDataset(env)

# Define loss criterion
criterion = {
    "x0_loss": PoseDistanceLoss(),
    "x1_loss": PoseDistanceLoss(),
}

# Define params to pass to training
params = {
    "camera_name": args.camera_name,
    "noise_scale": args.noise_scale
}

# Check if CUDA is available
use_cuda = torch.cuda.is_available()
device = "cuda:0" if use_cuda else "cpu"

# Global vars
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
best_trial_return = np.inf
log_num = 0


# Objective function
def objective(trial):
    # Global var for saving best trial run
    global best_trial_return
    global log_num

    # Algorithm hyperparams
    latent_dim = trial.suggest_categorical('latent_dim', [256, 512, 1024])
    sequence_length = trial.suggest_categorical('sequence_length', [10, 20, 50])
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-1)
    if args.model == 'naive':
        hidden_dim_start = trial.suggest_categorical('hidden_dim_start', [128, 256, 512])
        num_hidden_layers = trial.suggest_categorical('num_hidden_layers', [2, 3, 4])

        # Create hidden layers
        hidden_layers = []
        for i in range(num_hidden_layers):
            hidden_layers.append(hidden_dim_start // (2**i))
        model = NaiveEndEffectorStateEstimator(
            hidden_dims_pre_measurement=hidden_layers,
            hidden_dims_post_measurement=hidden_layers,
            num_resnet_layers=args.num_resnet_layers,
            latent_dim=latent_dim
        )
    elif args.model == 'td':
        lstm_h_dim = trial.suggest_categorical('lstm_h_dim', [128, 256, 512])
        model = TemporallyDependentStateEstimator(
            hidden_dim_pre_measurement=lstm_h_dim,
            hidden_dim_post_measurement=lstm_h_dim,
            num_resnet_layers=args.num_resnet_layers,
            latent_dim=latent_dim,
            sequence_length=sequence_length,
        )
    else:
        model = None

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Run experiment
    _, best_val_err = train(
        model=model,
        dataset=dataset,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.n_epochs,
        num_train_episodes_per_epoch=args.n_episodes_per_epoch,
        num_val_episodes_per_epoch=args.n_val_episodes_per_epoch,
        params=params,
        device=device,
        save_model=False,
        logging=False
    )

    # If this trial output is better than the previous, replace the "best" we have so far
    if best_val_err < best_trial_return:
        # Update best and notify user
        best_trial_return = best_val_err
        print("New best trial return = {}".format(best_val_err))

    # Lastly, return the trial output evaluation return
    return best_val_err


if __name__ == '__main__':
    # Setup study to run via optuna
    study = optuna.create_study(direction='minimize')

    # Execute study
    study.optimize(objective, n_trials=args.n_trials)

    print('Number of finished trials: ', len(study.trials))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))