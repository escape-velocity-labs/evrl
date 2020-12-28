"""

This module contains classes and functions that describe different
reinforcement learning (RL) policies.

"""

import torch
from torch.nn.functional import softmax


def get_e(episode, max_e: float, min_e: float, decay_episodes: int, decay: str) -> float:
    """

    Computes the level of exploration that the agent should operate under.

    Args:
        episode: the episode played by the agent.

        max_e: the maximum level of exploration during training.

        min_e: the minimum level of exploration during training.

        decay_episodes: the number of episodes that it will take for the exploration
        level to decay to the minimum value.

        decay: the decay scheme (how exploration changes over time).

    Returns: a float 0 <= e <= 1 representing the current level of exploration.

    """
    if decay == 'linear':
        epsilon = max_e - (max_e - min_e) * min(episode / decay_episodes, 1)

    elif decay == 'exponential':
        rate = torch.exp(torch.log(torch.Tensor(0.01)) / decay_episodes)
        epsilon = min_e + (max_e - min_e) * rate ** episode

    else:
        raise ValueError("Decay scheme not supported.")

    return epsilon


class GreedyPolicy:
    """

    A policy that selects the action with the highest value, whether it is
    a Q-Value or an action probability in the case of gradient policies.

    """

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        max_values = (values == values.max()).double()
        return max_values.multinomial(1)


class MultinomialPolicy:
    """

    A policy that selects a actions by sampling a multinomial
    probability distribution parameterized by the values on which
    it is called (Works with both Q-values and action probabilities).

    """

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        if values.sum() != 1.:
            values = softmax(values, dim=-1)
        return values.multinomial(1)


class ContinuousGreedyPolicy:
    """

    A policy that selects the most probable continuous actions after
    clamping them to the valid range.

    """

    def __init__(self, min_value: torch.Tensor, max_value: torch.Tensor) -> None:
        """

        Args:
            min_value: an array with the lower bounds of each dimension of
            the action space.

            max_value: an array with the higher bounds of each dimension of
            the action space.

        """

        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        clipped_actions = torch.max(torch.min(values, self.max_value), self.min_value)
        return clipped_actions


class EpsilonGreedyPolicy:
    """

    A policy that selects a random action with probability e and the
    action with the highest value with probability (1 - e).

    """

    def __init__(self, max_e: float = 1., min_e: float = 0.,
                 decay_episodes: int = 10000, decay: str = 'linear') -> None:
        """

        Args:
            max_e: the maximum level of exploration during training.

            min_e: the minimum level of exploration during training.

            decay_episodes: the number of episodes that it will take for the exploration
            level to decay to the minimum value.

            decay: the decay scheme (how exploration changes over time).

        """

        assert 0 <= min_e <= max_e <= 1
        assert decay_episodes >= 0
        assert decay in {'linear', 'exponential'}

        self.decay = decay
        self.max_e = max_e
        self.min_e = min_e
        self.decay_episodes = decay_episodes

    def __call__(self, values: torch.Tensor, episode: int) -> torch.Tensor:
        epsilon = get_e(episode, self.max_e, self.min_e, self.decay_episodes, self.decay)

        if torch.rand(1) < epsilon:
            return torch.randint(values.shape[-1], size=(1, 1))

        max_values = (values == values.max()).double()
        return max_values.multinomial(1)


class EpsilonMultinomialPolicy:
    """

    A policy that picks a random action with probability e and
    selects from a multinomial parameterized by the values on
    which it is called with probability (1 - e).

    """

    def __init__(self, max_e: float = 1., min_e: float = 0.,
                 decay_episodes: int = 10000, decay: str = 'linear') -> None:
        """

        Args:
            max_e: the maximum level of exploration during training.

            min_e: the minimum level of exploration during training.

            decay_episodes: the number of episodes that it will take for the exploration
            level to decay to the minimum value.

            decay: the decay scheme (how exploration changes over time).

        """

        assert 0 <= min_e <= max_e <= 1
        assert decay_episodes >= 0
        assert decay in {'linear', 'exponential'}

        self.decay = decay
        self.max_e = max_e
        self.min_e = min_e
        self.decay_episodes = decay_episodes

    def __call__(self, values: torch.Tensor, episode: int) -> torch.Tensor:
        epsilon = get_e(episode, self.max_e, self.min_e, self.decay_episodes, self.decay)

        if torch.rand(1) < epsilon:
            return torch.randint(values.shape[-1], size=(1, 1))

        return values.multinomial(1)


class ContinuousEpsilonGreedyPolicy:
    """

    A policy that applies gaussian noise with mu=0 and std= e * (valid range)
    where e is an exploration constant to be annealed during training.

    """

    def __init__(self, min_value: torch.Tensor, max_value: torch.Tensor,
                 max_e: float = 1., min_e: float = 0., decay_episodes: int = 10000,
                 decay: str = 'linear') -> None:
        """

        Args:
            min_value: an array with the lower bounds of each dimension of
            the action space.

            max_value: an array with the higher bounds of each dimension of
            the action space.

            max_e: the maximum level of exploration during training.

            min_e: the minimum level of exploration during training.

            decay_episodes: the number of episodes that it will take for the exploration
            level to decay to the minimum value.

            decay: the decay scheme (how exploration changes over time).
        """

        assert 0 <= min_e <= max_e <= 1
        assert decay_episodes >= 0
        assert decay in {'linear', 'exponential'}

        self.min_value = min_value
        self.max_value = max_value
        self.max_e = max_e
        self.min_e = min_e
        self.decay_episodes = decay_episodes
        self.decay = decay

    def __call__(self, values: torch.Tensor, episode: int) -> torch.Tensor:
        epsilon = get_e(episode, self.max_e, self.min_e, self.decay_episodes, self.decay)

        # Standard deviation based on parameter e.
        sigma = epsilon * (self.max_value - self.min_value) / 2
        noise = torch.randn(values.size()) * sigma
        values = values + noise
        clipped_actions = torch.max(torch.min(values, self.max_value), self.min_value)
        return clipped_actions
