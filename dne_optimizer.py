import gym
from typing import Tuple, List
from multiprocessing import Pool, cpu_count

import numpy as np

from utils import PolicyInterface, ObsNormalizer
from evolution_strategies import ESInterface


class DeepEvolution:

    def __init__(self, env_factory: function,
                 policy_factory: function,
                 evolution_strategy: ESInterface,
                 evaluation_function: function = None):
        self._env_factory: function = env_factory
        self._model_factory: function = policy_factory
        if evaluation_function is None:
            self.evaluation_function = self.default_eval_function
        else:
            self.evaluation_function = evaluation_function

        self.evolution_strategy: ESInterface = evolution_strategy
        self.normalizer: ObsNormalizer = ObsNormalizer(env_factory, n_samples=1000)
        self.generation: List[PolicyInterface] = [policy_factory()]
        # something_callback: function

    def default_eval_function(self, policy: PolicyInterface, times=1,) -> Tuple[int, PolicyInterface]:
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
        with Pool(cpu_count()) as p:
            # List[Tuple[float, PolicyInterface]]
            current_gen = p.starmap(self.evaluation_function, self.generation)
        self.generation = self.evolution_strategy(current_gen)

