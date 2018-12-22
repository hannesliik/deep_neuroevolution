from abc import ABC, abstractmethod
from typing import List, Tuple, Callable
from multiprocessing import Pool, cpu_count

from gym import Env

import numpy as np

from utils import Policy, ObsNormalizer


class Evaluator(ABC):
    """
    Evaluator, when called, should take in a list of policies and return a
    list of tuples (score, policy) and the best policy
    """
    @abstractmethod
    def __call__(self, policies: List[Policy]) -> Tuple[List[Tuple[float, Policy]], Policy]:
        pass


class EnvEvaluator(Evaluator):
    def __init__(self, env_factory: Callable):
        self.env: Env = env_factory()
        self.normalizer = ObsNormalizer(env_factory)

    def __call__(self, policies: List[Policy], times=1) -> Tuple[List[Tuple[float, Policy]], Policy]:
        results: List[Tuple[float, Policy]] = []
        for policy in policies:
            rewards = []
            for _ in range(times):
                obs = self.env.reset()
                done = False
                total_reward = 0
                while not done:
                    obs = self.normalizer.normalize(obs)
                    action = policy(obs)
                    obs, reward, done, _ = self.env.step(action)
                    total_reward += reward
                rewards.append(total_reward)
            results.append((int(np.mean(rewards)), policy))
        results = sorted(results, key=lambda x: x[0], reverse=True)
        return results, results[0][1]


class ParallelEnvEvaluator(Evaluator):
    def __init__(self, env_factory: Callable):
        self.env_factory = env_factory
        self.normalizer = ObsNormalizer(env_factory)

    def __call__(self, policies: List[Policy], times=1) -> Tuple[List[Tuple[float, Policy]], Policy]:
        with Pool(cpu_count()) as p:
            # List[Tuple[float, PolicyInterface]]
            results = p.starmap(self._eval_policy, [(policy, times) for policy in policies])
        results = sorted(results, key=lambda x: x[0], reverse=True)
        return results, results[0][1]

    def _eval_policy(self, policy: Policy, times=1):
        env: Env = self.env_factory()
        rewards = []
        for _ in range(times):
            obs = env.reset()
            done = False
            total_reward = 0
            while not done:
                obs = self.normalizer.normalize(obs)
                action = policy(obs)
                obs, reward, done, _ = env.step(action)
                total_reward += reward
            rewards.append(total_reward)
        return int(np.mean(rewards)), policy
