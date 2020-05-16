from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import torch


class MultiEpisodeDataset(Dataset):
    """
    Class for processing multiple episodes as a dataset
    """

    def __init__(self, env, custom_transform=None):
        """
        Initializes custom dataset class

        Note: @self.data (dict): Dict holding all observations from a given episode. Format is as follows:
            'imgs': Camera observations from that episode
            'measurement_self': Measurement of self eef state
            'true_self': True self eef state
            'true_other': True other eef state

        Args:
            env (MujocoEnv): Env to grab online sim data from
            custom_transform (Transform): (Optional) Custom transform to be applied to image observations. Default is
                None, which results in default transform being applied (Normalization + Scaling)
        """
        self.data = None
        self.env = env

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

    def __len__(self):
        return self.data["measurement_self"].size(1)

    def __getitem__(self, index):
        img = self.data['imgs'][:, index, :, :, :]      # imgs shape (n_ep, ep_length, C, H, W)
        x0bar = self.data['measurement_self'][:, index, :]  # measurement_self shape (n_ep, ep_length, x_dim)
        x0 = self.data['true_self'][:, index, :]    # true_self shape (n_ep, ep_length, x_dim)
        x1 = self.data['true_other'][:, index, :]   # true_other shape (n_ep, ep_length, x_dim)

        return img, x0bar, x0, x1

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
            'measurement_self': [],
            'true_self': [],
            'true_other': []
        }

        # Get action limits
        low, high = self.env.action_spec

        for i in range(num_episodes):
            # Reset env
            self.env.reset()

            # Reset auto variables
            imgs = []
            measurement_self = []
            true_self = []
            true_other = []
            done = False

            print("Running episode {} of {}...".format(i+1, num_episodes))

            # Start grabbing data after each step
            while not done:
                # Grab random action for entire action space
                action = np.random.uniform(low, high)
                obs, reward, done, _ = self.env.step(action)

                # Grab appropriate info as needed

                # Need to preprocess image first before appending
                img = obs[camera_name + "_image"]
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
                true_other.append(np.concatenate([obs["robot1_eef_pos"], obs["robot1_eef_quat"]]))

            # After episode completes, add to new_data
            new_data["imgs"].append(torch.stack(imgs))
            new_data["measurement_self"].append(measurement_self)
            new_data["true_self"].append(true_self)
            new_data["true_other"].append(true_other)

        # Finally convert new data to Tensors
        new_data["imgs"] = torch.stack(new_data["imgs"])
        new_data["measurement_self"] = torch.tensor(new_data["measurement_self"], dtype=torch.float, requires_grad=False)
        new_data["true_self"] = torch.tensor(new_data["true_self"], dtype=torch.float, requires_grad=False)
        new_data["true_other"] = torch.tensor(new_data["true_other"], dtype=torch.float, requires_grad=False)

        # Now save the new_data as self.data
        self.data = new_data
