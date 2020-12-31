"""

Utility methods and classes.

"""

from typing import Iterable, Dict
import numpy as np
import torch
import gym
from evrl.common.envs import BufferEnv

EPS = torch.finfo(torch.float32).eps


def generate_random_baseline(env: BufferEnv, episodes: int) -> Dict[str, Iterable[torch.Tensor]]:
    """

    Runs the given number of episodes sampling random actions from the
    environment's action space and returns a dictionary with the experiment
    results.

    Args:
        env: the task to be solved.

        episodes: the number of times the agent tries to solve the task.

    Returns: a dictionary of result statistics.

    """
    durations = []
    returns = []
    for _ in range(1, episodes + 1):
        env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step(env.action_space.sample())
        durations.append(env.duration)
        returns.append(env.rewards.sum())
    return {'durations': durations, 'returns': returns}


def test_agent(env: BufferEnv, agent: torch.nn.Module, render=False, episodes=1):
    """

    Test the performance of an arbitrary agent trying to solve a task.

    Args:
        env: the task to be solved.

        agent: the entity that takes actions in the environment.

        render: whether the experiment should be rendered.

        episodes: the number of times the agent tries to solve the task.

    Returns: a dictionary of result statistics.

    """
    durations = []
    returns = []
    for _ in range(1, episodes + 1):
        state = env.reset()
        done = False
        while not done:
            if render:
                env.render(mode='human')
            action = agent(state).argmax()
            new_state, _, done, _ = env.step(action)
            state = new_state
        durations.append(env.duration)
        returns.append(env.rewards.sum())
    return {'durations': durations, 'returns': returns}


def discount_rewards(rewards: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
    """

    Discounts the rewards in the input tensor.

    Args:
        rewards: the rewards to be discounted.

        gamma: the discount factor.

    Returns:
        A tensor of discounted rewards.

    """
    gammas = gamma ** torch.arange(len(rewards)).view(rewards.size())
    discounted = torch.cumsum(rewards * gammas, dim=0).flip(dims=(0,))
    return discounted


def perform_update(optimizer: torch.optim.Optimizer,
                   parameters: Iterable[torch.Tensor], loss: torch.Tensor) -> None:
    """

    Clips the gradient tensors by a global norm and performs a step of
    gradient descent.

    Args:
        optimizer: the update rule to be used for gradient descent.

        parameters: the paramerters to be modified by gradient descent.

        loss: tensor to be minimized by gradient descent.

    Returns:
        None.

    """
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters, 40)
    optimizer.step()


def polyak_average(model_from: torch.nn.Module,
                   model_to: torch.nn.Module, rho: float = 0.01) -> None:
    """

    Args:
        model_from: the source of the parameters to be copied.

        model_to: the target holding the parameters to be modified.

        rho: a parameter describing the weight of the source parameters
        in the new values.

    Returns:
        None.
    """
    for src_param, dest_param in zip(model_from.parameters(), model_to.parameters()):
        dest_param.data = rho * src_param.data + (1 - rho) * dest_param.data


def seed_everything(env: gym.Env, seed: int = 42) -> None:
    """

    Seeds all the sources of randomness so that experiments are reproducible.

    Args:
        env: the environment to be seeded.
        seed: an integer seed.

    Returns:
        None.

    """
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_deterministic(True)


def entropy(values: torch.Tensor) -> torch.Tensor:
    """

    Compute the entropy of a Categorical distribution parameterized
    by the input tensor 'values'.

    Args:
        values: logits or probs of the Categorical distribution.

    Returns:
        A tensor with the entropy values.

    """

    values = values / values.sum(dim=-1)
    values = values.clamp(min=EPS)
    values = values * values.log()
    values = - values.sum(dim=-1, keepdim=True)
    return values
