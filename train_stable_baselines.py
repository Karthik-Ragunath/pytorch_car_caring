import argparse

import numpy as np

import gym
from stable_baselines3 import PPO
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from utils import DrawLine
import logging
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.callbacks import BaseCallback
import os

parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 8)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--vis', action='store_true', help='use visdom')
parser.add_argument(
    '--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')
parser.add_argument("--device_id", "-dev", type=int, default=0, required=False)
parser.add_argument("--log_seed", type=int, default=0, required=False)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device(f"cuda:{args.device_id}" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

transition = np.dtype([('s', np.float64, (args.img_stack, 96, 96)), ('a', np.float64, (3,)), ('a_logp', np.float64),
                       ('r', np.float64), ('s_', np.float64, (args.img_stack, 96, 96))])

LOGGER= logging.getLogger()
LOGGER.setLevel(logging.DEBUG) # or whatever
handler = logging.FileHandler(f"ppo_logger_{args.log_seed}.log", 'w', 'utf-8')
formatter = logging.Formatter('%(name)s %(message)s')
handler.setFormatter(formatter)
LOGGER.addHandler(handler)

# class Env(gym.Env):
class Env(gym.Env):
    """
    Environment wrapper for CarRacing 
    """
    metadata = {"render.modes": ["human", "rgb_array"]}
    def __init__(self):
        # super(Env, self).__init__()
        if args.render:
            self.env = gym.make('CarRacing-v0')
            self.env.render("rgb_array")
        else:
            self.env = gym.make('CarRacing-v0')
        # self.env.seed(args.seed)
        self.reward_threshold = self.env.spec.reward_threshold
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb = self.env.reset()
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [img_gray] * args.img_stack  # four frames for decision
        return np.array(self.stack).transpose(1,2,0)

    def step(self, action):
        total_reward = 0
        for i in range(args.action_repeat):
            img_rgb, reward, die, info = self.env.step(action)
            # don't penalize "die state"
            if die:
                reward += 100
            # green penalty
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                reward -= 0.05
            total_reward += reward
            # if no reward recently, end the episode
            done = True if self.av_r(reward) <= -0.1 else False
            if done or die:
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == args.img_stack
        return np.array(self.stack).transpose(1,2,0), total_reward, die or done, info

    def render(self, *arg):
        self.env.render(*arg)

    def close(self):
        self.env.close()

    @staticmethod
    def rgb2gray(rgb, norm=True):
        # rgb image -> gray [0, 1]
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        return gray

    @staticmethod
    def reward_memory():
        # record reward for last 100 steps
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = args.img_stack
        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(n_input_channels, 8, kernel_size=4, stride=2),
            nn.ReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),  # activation
            nn.Conv2d(128, features_dim, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),  # activation
        )  # output shape (256, 1, 1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.cnn_base(observations)

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
)

class Net(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self):
        super(Net, self).__init__()
        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(args.img_stack, 8, kernel_size=4, stride=2),
            nn.ReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),  # activation
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),  # activation
        )  # output shape (256, 1, 1)
        self.v = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, 1))
        self.fc = nn.Sequential(nn.Linear(256, 100), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return (alpha, beta), v

class CustomCallBack(BaseCallback):
    """Custom CallBack Class."""
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(CustomCallBack, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        """Init callback function."""
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _on_training_start(self) -> None:
        pass

    def _on_training_end(self) -> None:
        pass

    def _on_step(self) -> bool:
        """Return False to abort training early."""
        # self.locals - gives local variables in a dictionary
        return True

# -----------------------------------------------------------------------------------
class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = args.img_stack
        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(n_input_channels, 8, kernel_size=4, stride=2),
            nn.ReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),  # activation
            nn.Conv2d(128, features_dim, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),  # activation
            nn.Flatten(),
        )  # output shape (256, 1, 1)
        self.linear = nn.Sequential(nn.Linear(features_dim, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn_base(observations))


class CustomNetwork(nn.Module):
    """
    Actor Critic Network For PPO.
    """
    def __init__(self, features_dim):
        super(CustomNetwork, self).__init__()
        self.latent_dim_pi = 100
        self.latent_dim_vf = 100
        self.v_latent = nn.Sequential(nn.Linear(256, 100), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(256, 100), nn.ReLU())
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1
        return (alpha, beta), v
        """
        # x = self.cnn_base(x)
        # x = x.view(-1, 256)
        # print('*'*50, x.shape, '*'*50)
        return self.forward_actor(x), self.forward_critic(x)

    def forward_actor(self, x: torch.tensor) -> torch.tensor:
        return self.fc(x)

    def forward_critic(self, x: torch.tensor) -> torch.tensor:
        return self.v_latent(x)

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        # use_sde: bool,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # use_sde,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

if __name__ == "__main__":
    # env_id = "CarRacing-v2"
    env = Env()
    '''
    model = PPO(
        CustomActorCriticPolicy(
            observation_space=env.env.observation_space, 
            action_space=env.env.action_space,
            lr_schedule=linear_schedule(1e-3),
            use_sde=False
        ), env, verbose=1)
        # ), env, use_sde=False, verbose=1)
    '''
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )
    model = PPO(CustomActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=10000)
    if os.path.exists('sb3_files'):
        os.makedirs('sb3_files')
    
    entire_model_save_path = os.path.join('sb3_files', 'entire_model')
    if not os.path.exists(entire_model_save_path):
        os.makedirs(entire_model_save_path)       
    model.save(entire_model_save_path)

    state_dict_save_path = os.path.join('sb3_files', 'state_dict')
    if not os.path.exists(state_dict_save_path):
        os.makedirs(state_dict_save_path)
    torch.save(model.state_dict(), os.path.join(state_dict_save_path, 'ppo_net_params_model_trained.pkl'))
