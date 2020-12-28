"""

This module contains wrappers and convenience functions to simplify
working with gym environments of different kinds.

"""

from typing import Tuple, Dict, Union
import gym
import numpy as np
import torch
from gym.wrappers import AtariPreprocessing, FrameStack
from gym import spaces


class BufferEnv(gym.Wrapper):
    """

    Wrapper class that prepares a gym.Env instance to be used with PyTorch
    as well as to store all transitions of the episode (until it is reset).

    This class keeps track of the following elements of the episode:

        - Observations.

        - Actions taken by the agent.

        - Rewards.

        - Done signal (whether the episode is over at time step t).

        - Episode number.

        - Duration of the episode.

        - Additional information (dict) provided by the wrapped environment class.

        - (Optionally) Frames rendered by the environment (as torch.Tensor).

    """

    def __init__(self, env: gym.Env, render_every: int = 50) -> None:
        """

        Initialize the wrapper.

        Args:
            env: gym.Env to be wrapped by this class.
            render_every: frequency at which the environment should render an episode.
            Default is 50.

        """

        super().__init__(env)
        self.render_every = render_every
        self.episode = 0
        if isinstance(env.observation_space, spaces.dict.Dict):
            self._observations = {'observation': [], 'achieved_goal': [], 'desired_goal': []}
        else:
            self._observations = []
        self._actions = []
        self._dones = []
        self._rewards = []
        self._frames = []
        self.infos = []
        self.duration = 0

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """

        Take an action in the environment and observe the next transition.
        Prepare the variables received from the environment to be used by
        the PyTorch agent.

        Args:
            action: torch.Tensor that can be scalar or n-dimensional,
            integer or float depending on the action space.

        Returns: The new observation, reward, done signal and additional info as a tuple.

        """

        # Strip the extra dimensions and convert the action to a numpy array for
        # compatibility with the wrapped gym.Env.
        observation, reward, done, info = super().step(action.squeeze().numpy())

        # Add new dimension for shape compatibility and convert the
        # item to a torch.Tensor.
        reward = torch.FloatTensor([[reward]])
        done = torch.BoolTensor([[done]])

        # Goal oriented tasks keep the observations in a dictionary, where the observation, the goal
        # and the current goal achieved are stored in different keys.
        if isinstance(observation, dict):
            for key, value in observation.items():
                value = torch.from_numpy([value])
                self._observations[key].append(value)
                observation[key] = value
        else:
            observation = torch.tensor([observation])
            self._observations.append(observation)

        # Update the environment buffers with the new transition.
        self._actions.append(action)
        self._rewards.append(reward)
        self._dones.append(done)
        self.infos.append(info)
        self.duration += 1

        # If the environment should render this episode, keep
        # the frames in a buffer.
        if self.episode % self.render_every == 0:
            frame = self.render(mode='rgb_array')
            # Tensorboard accepts frames of shape (T x C x H x W), so we need
            # to convert the frame to channels-first and add an extra dimension
            # at position 0.
            frame = np.rollaxis(frame, 2)[np.newaxis, :, :, :]
            frame = torch.from_numpy(frame.copy())
            self._frames.append(frame)

        return observation, reward, done, info

    def reset(self, **kwargs) -> torch.Tensor:
        """

        Resets the wrapped environment and clears all the buffers.

        Args:
            **kwargs: Keyword arguments accepted by the wrapped gym.Env object.

        Returns: A torch.Tensor representing the initial observation of the new episode.

        """

        observation = super().reset(**kwargs)

        # Clear buffers.
        self.reset_buffers()
        self.episode += 1

        # Prepare the initial observation for use with PyTorch.
        if isinstance(observation, dict):
            for key, value in observation.items():
                value = torch.tensor([value])
                self._observations[key].append(value)
                observation[key] = value
        else:
            observation = torch.tensor([observation])
            self._observations.append(observation)
        return observation

    def reset_buffers(self) -> None:
        """

        Reset the buffers that contain the transitions.

        Returns: None.

        """
        if isinstance(self.observation_space, spaces.dict.Dict):
            for key in self._observations:
                self._observations[key].clear()
        else:
            self._observations.clear()
        self._actions.clear()
        self._dones.clear()
        self._rewards.clear()
        self._frames.clear()
        self.infos.clear()
        self.duration = 0

    def compute_reward(self, achieved_goal, desired_goal, info) -> float:
        """

        Computes the reward to be awarded to the agent for achieving goal 'a',
        when the desired goal is 'b'.

        Args:
            achieved_goal: the goal attained by the agent.

            desired_goal: the goal state that the agent must attain to
            solve the environment.

            info: a dictionary containing additional information
            about the environment.

        Returns: a reward signal for the agent.

        """

        reward = super().compute_reward(achieved_goal, desired_goal, info)
        reward = torch.from_numpy(reward).unsqueeze(1)
        return reward

    @property
    def states(self) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        if isinstance(self.observation_space, spaces.dict.Dict):
            obs = {}
            for key in self._observations:
                obs[key] = torch.cat(self._observations[key])
                return obs
        return torch.cat(self._observations)

    @property
    def actions(self) -> torch.Tensor:
        return torch.cat(self._actions)

    @property
    def rewards(self) -> torch.Tensor:
        return torch.cat(self._rewards)

    @property
    def dones(self) -> torch.Tensor:
        return torch.cat(self._dones)

    @property
    def frames(self) -> torch.Tensor:
        return torch.cat(self._frames)


class NormalizeObservationWrapper(gym.ObservationWrapper):
    """

    This wrapper maps the pixel observations of the original gym.Env to
    the range [0, 1].

    """

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """

        Modify the original observation and return the result.

        Args:
            observation: the original array representing a pixel observation
            from the environment.

        Returns: an array of the same shape, whose values are in the [0, 1] range.

        """

        return observation / 255.


# TODO: This implementation is broken.
class TileCodingEnv(gym.ObservationWrapper):
    """

    """

    def __init__(self, env: gym.Env, n_tilings: int, bins: int, clip_inf: float = None) -> None:
        """

        Args:
            env:
            n_tilings:
            bins:
            clip_inf:
        """
        super().__init__(env)
        low = self.observation_space.low
        high = self.observation_space.high
        if clip_inf is not None:
            low = np.clip(low, a_min=-1 * clip_inf, a_max=None)
            high = np.clip(high, a_min=None, a_max=clip_inf)
        ranges = high - low
        max_offsets = ranges / bins
        tilings = []
        for _ in range(n_tilings):
            dim_tiles = []
            for i, _ in enumerate(low):
                lower_bound = low[i]
                higher_bound = high[i]
                offset = lower_bound + np.random.rand(1) * max_offsets[i]
                tile = np.linspace(start=lower_bound,
                                   stop=higher_bound, num=bins)[:-1] + offset
                dim_tiles.append(tile)
            tilings.append(dim_tiles)

        self.tilings = tilings

    def observation(self, observation):
        """

        Args:
            observation:

        Returns:

        """
        feat_codings = []
        for tiling in self.tilings:
            feat_coding = []
            for i in range(len(self.low)):
                feat_i = observation[i]
                tiling_i = tiling[i]  # tiling on that dimension
                coding_i = np.digitize(feat_i, tiling_i)
                feat_coding.append(coding_i)
            feat_codings.append(feat_coding)
        return np.array(feat_codings)


class DiscretizedEnv(gym.ObservationWrapper):
    """

    Applies domain discretization to the observations of the wrapped
    environment.

    """

    def __init__(self, env: gym.Env, bins: np.array,
                 clip_max: np.array = None, clip_min: np.array = None) -> None:
        """

        Modify the observation space to reflect the discretization
        process and prepare the grid that will be used to transform
        the observations.

        Args:
            env: the wrapped environment.

            bins: the number of buckets to which each dimension of the
            observation will be mapped.

            clip_max: an array describing the maximum values that each
            dimension of the observation can take.

            clip_min: an array describing the minimum values that each
            dimension of the observation can take.

        """

        super().__init__(env)
        low = self.observation_space.low
        high = self.observation_space.high

        if clip_max is not None:
            high = np.minimum(high, clip_max)
        if clip_min is not None:
            low = np.maximum(low, clip_min)

        self.bins = bins
        self.grid = [np.linspace(start=low, stop=high, num=bin_n + 1)[1:-1]
                     for low, high, bin_n in zip(low, high, bins)]
        self.total_bins = int(np.prod(bins))
        self.observation_space = gym.spaces.Box(
            low=np.zeros((self.total_bins,)),
            high=np.ones((self.total_bins,)),
            shape=(self.total_bins,),
            dtype=np.float64)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """

        Map the continuous observation to a discrete range.

        Args:
            observation: the original observation returned by the
            wrapped environment.

        Returns: the discretized observation.

        """

        one_hot = np.zeros(shape=tuple(self.bins))
        idx = tuple(np.digitize(value, bounds) for value, bounds in zip(observation, self.grid))
        one_hot[idx] = 1
        flat_one_hot = one_hot.flatten()
        return flat_one_hot


def make_env(env_name: str, render_every: int = 50) -> gym.Env:
    """

    Make an RL environment and prepare it to be used
    with PyTorch.

    Args:
        env_name: the name of the environment to be created.

        render_every: frequency with which the episode should be rendered.

    Returns: the RL environment.

    """

    env = gym.make(env_name)
    env = BufferEnv(env, render_every=render_every)
    return env


def make_discrete_env(env_name: str, bins: np.ndarray, clip_max: np.ndarray = None,
                      clip_min: np.ndarray = None) -> gym.Env:
    """

    Creates an RL environment and maps its continuous observations into
    discrete ranges to discretize them.

    Args:
        env_name: the name of the environment to be created.

        bins: number of bins over which to discretize each variable.

        clip_max: an array of max values for each continuous variable.
        Observations with values above this limit will be clipped.

        clip_min: an array of min values for each continuous variable.
        Observations with values below this limit will be clipped.

    Returns: an environment with a discrete observation space.

    """

    env = gym.make(env_name)
    env = DiscretizedEnv(env, bins=bins, clip_max=clip_max, clip_min=clip_min)
    env = BufferEnv(env)
    return env


def make_atari(env_name: str) -> gym.Env:
    """

    Creates an RL environment based on an ATARI console game and applies
    the necessary wrappers to simplify working with it.

    Args:
        env_name: the name of the environment to be created.

    Returns: the ATARI RL environment.

    """

    env = gym.make(env_name)
    env = AtariPreprocessing(env)
    env = FrameStack(env, num_stack=4)
    env = NormalizeObservationWrapper(env)
    env = BufferEnv(env)
    return env
