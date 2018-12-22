import gym
from typing import Tuple, List

import numpy as np

from utils import PolicyInterface, ObsNormalizer
from evolution_strategies import ESInterface


class DeepEvolution:

    def __init__(self, env_factory: function, model_factory: function,
                 evaluation_function: function, evolution_strategy: function):
        self._env_factory = env_factory
        self._model_factory = model_factory
        self.evaluation_function = evaluation_function
        self.evolution_strategy = evolution_strategy
        self.normalizer = ObsNormalizer(env_factory, n_samples=1000)
        # something_callback: function

    def default_eval_function(self, policy: PolicyInterface, times=1, ) -> Tuple[int, PolicyInterface]:
        env: gym.Env = self._env_factory()
        rewards = []
        for _ in range(times):
            obs = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = policy(obs)
                obs, reward, done, _ = env.step(action)
                obs = self.normalizer.normalize(obs)
                total_reward += reward
            rewards.append(total_reward)
        return int(np.mean(rewards)), policy

    def train_generation(self):
        pass
