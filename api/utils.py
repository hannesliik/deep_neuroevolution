import gym
from abc import ABC, abstractmethod
import numpy as np
from typing import Callable


class Policy(ABC):
    """
    An interface for policies. It should accept a numpy array as the observation
    and return a numpy array as the action
    """

    @abstractmethod
    def __call__(self, observation: np.ndarray) -> np.ndarray:
        """
        Given an observation, should return an action
        :param observation: the observation
        :return: the action/prediction
        """
        pass

    def set_up(self):
        """
        Stuff to do before evaluating (useful in the parallel evaluator) like move to GPU
        :return:
        """
        pass

    def teardown(self):
        """
        Stuff to do after evaluation
        :return:
        """
        pass


class ObsNormalizer:
    """
    A class to remember observation statistics
    and normalize future observations
    """
    EPSILON = 1e-7

    def __init__(self, env_factory: Callable, n_samples=1000):
        env: gym.Env = env_factory()
        self.means, self.stds = self.gen_statistics(env, n_samples)

    def normalize(self, obs: np.ndarray):
        return (obs - self.means) / (self.stds + ObsNormalizer.EPSILON)

    def unnormalize(self, obs: np.ndarray):
        return (obs + self.means) * (self.stds + ObsNormalizer.EPSILON)

    @staticmethod
    def gen_statistics(env, n):
        obs = env.reset()
        obs_dataset = [obs]
        for _ in range(n):
            action = env.action_space.sample()
            obs_dataset.append(obs)
            obs, reward, done = env.step(action)[:3]
            if done:
                obs = env.reset()
        obs_dataset = np.array(obs_dataset)
        means = np.mean(obs_dataset, axis=0)
        stds = np.std(obs_dataset, axis=0)
        return means, stds
