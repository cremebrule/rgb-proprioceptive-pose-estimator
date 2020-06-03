from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import torch


MOTIONS = {"random", "up"}


class MultiEpisodeDataset(Dataset):
    """
    Class for processing multiple episodes as a dataset
    """

    def __init__(self, env, use_depth=False, obj_name=None, motion="random", custom_transform=None):
        """
        Initializes custom dataset class

        Note: @self.data (dict): Dict holding all observations from a given episode. Format is as follows:
            'imgs': Camera observations from that episode
            'measurement_self': Measurement of self eef state
            'true_self': True self eef state
            'true_other': True other eef state

        Args:
            env (MujocoEnv): Env to grab online sim data from
            use_depth (bool): Whether to use depth observations or not
            obj_name (str): Name of object to grab pose values from environment
            motion (str): Type of motion to use. Supported options are "random" and "up" currently
            custom_transform (Transform): (Optional) Custom transform to be applied to image observations. Default is
                None, which results in default transform being applied (Normalization + Scaling)
        """
        self.data = None
        self.env = env
        self.obj_name = obj_name
        self.use_depth = use_depth
        self.is_two_arm = "TwoArm" in str(type(self.env))
        if motion in MOTIONS:
            self.motion = motion
        else:
            raise ValueError("Invalid motion specified. {} supported, {} requested.".format(MOTIONS, motion))

        if custom_transform:
            self.transform = custom_transform
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.depth_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return self.data["measurement_self"].size(1)

    def __getitem__(self, index):
        img = self.data['imgs'][:, index, :, :, :]                              # imgs shape (n_ep, ep_length, C, H, W)
        depth = self.data['depths'][:, index, :, :, :] if self.use_depth else torch.empty(*img.shape)  # depths shape (n_ep, ep_length, 1, H, W)
        x0bar = self.data['measurement_self'][:, index, :]                      # measurement_self shape (n_ep, ep_length, x_dim)
        x0 = self.data['true_self'][:, index, :]                                # true_self shape (n_ep, ep_length, x_dim)
        x1 = self.data['true_other'][:, index, :] if self.is_two_arm else torch.empty(*x0.shape)  # true_other shape (n_ep, ep_length, x_dim)
        obj = self.data['true_obj'][:, index, :] if self.obj_name is not None else torch.empty(*x0.shape)

        return img, depth, x0bar, x0, x1, obj

    def refresh_data(self, num_episodes, camera_name, noise_scale):
        """
        Refreshes data by drawing new data from self.env for the specified number of episodes

        Args:
            num_episodes (int): Number of episodes to run and grab data from
            camera_name (str): Name of camera from which to grab image observations from
            noise_scale (float): Scales the I noise covariance by this factor
        """
        # Create variable to hold new data
        new_data = {
            'imgs': [],
            'depths': [],
            'measurement_self': [],
            'true_self': [],
            'true_other': [],
            'true_obj': []
        }

        # Get action limits
        low, high = self.env.action_spec

        for i in range(num_episodes):
            # Reset env
            self.env.reset()

            # Reset auto variables
            imgs = []
            depths = []
            measurement_self = []
            true_self = []
            true_other = []
            true_obj = []
            sub_traj_steps = 10
            sub_traj_ct = 0
            steps = 0
            direction = 1
            action = np.random.uniform(low, high)
            done = False

            print("Running episode {} of {}...".format(i+1, num_episodes))

            # Start grabbing data after each step
            while not done:
                # Create action based on type specified
                if self.motion == "random":
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
                else:   # type "up"
                    # Move upwards
                    action = np.zeros(len(low))
                    action[2] = direction * high[2]
                    # Also check if we need to switch directions
                    if (steps + 1) % 20 == 0:
                        direction = -direction

                # Execute action
                obs, reward, done, _ = self.env.step(action)

                # Grab appropriate info as needed

                # Need to preprocess image first before appending
                img = obs[camera_name + "_image"]
                if self.use_depth:
                    depth = obs[camera_name + "_depth"]
                    depths.append(self.depth_transform(depth).float())
                #print(obs[camera_name + "_image"].shape)
                #img = Image.fromarray(np.uint8(np.transpose(obs[camera_name + "_image"], (2,0,1))))       # new shape should be (C, H, W)
                #print(img.shape)
                imgs.append(self.transform(img).float())
                x0 = np.concatenate([obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
                x0bar = x0 + np.random.multivariate_normal(mean=np.zeros(7), cov=np.eye(7) * noise_scale)
                # Renormalize the orientation part
                mag = np.linalg.norm(x0bar[3:])
                x0bar[3:] /= mag
                measurement_self.append(x0bar)
                true_self.append(x0)
                # Get second arm pose if environment has multiple arms
                if self.is_two_arm:
                    true_other.append(np.concatenate([obs["robot1_eef_pos"], obs["robot1_eef_quat"]]))
                if self.obj_name is not None:
                    # Get object observations if requested
                    true_obj.append(np.concatenate([obs[self.obj_name + "_pos"], obs[self.obj_name + "_quat"]]))

                # Increment step
                steps += 1

            # After episode completes, add to new_data
            new_data["imgs"].append(torch.stack(imgs))
            if self.use_depth:
                new_data["depths"].append(torch.stack(depths))
            new_data["measurement_self"].append(measurement_self)
            new_data["true_self"].append(true_self)
            if self.is_two_arm:
                new_data["true_other"].append(true_other)
            if self.obj_name is not None:
                new_data["true_obj"].append(true_obj)

        # Finally convert new data to Tensors
        new_data["imgs"] = torch.stack(new_data["imgs"])
        if self.use_depth:
            new_data["depths"] = torch.stack(new_data["depths"])
        new_data["measurement_self"] = torch.tensor(new_data["measurement_self"], dtype=torch.float, requires_grad=False)
        new_data["true_self"] = torch.tensor(new_data["true_self"], dtype=torch.float, requires_grad=False)
        if self.is_two_arm:
            new_data["true_other"] = torch.tensor(new_data["true_other"], dtype=torch.float, requires_grad=False)
        if self.obj_name is not None:
            new_data["true_obj"] = torch.tensor(new_data["true_obj"], dtype=torch.float, requires_grad=False)

        # Now save the new_data as self.data
        self.data = new_data
