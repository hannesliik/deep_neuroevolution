from abc import ABC, abstractmethod
from typing import List, Tuple
from multiprocessing import Pool, cpu_count

from gym import Env
import numpy as np

from utils import PolicyInterface, ObsNormalizer

class Evaluator(ABC):
    @abstractmethod
    def __call__(self, policies: List[PolicyInterface]) -> List[Tuple[float, PolicyInterface]]:
        pass

class EnvEvaluator(Evaluator):
    def __init__(self, env_factory: function):
        self.env: Env = env_factory()
        self.normalizer = ObsNormalizer(env_factory)

    def __call__(self, policies: List[PolicyInterface], times=1) -> List[Tuple[float, PolicyInterface]]:
        results: List[Tuple[float, PolicyInterface]] = []
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
        return results

class ParallelEnvEvaluator(Evaluator):
    def __init__(self, env_factory: function):
        self.env_factory = env_factory
        self.normalizer = ObsNormalizer(env_factory)

    def __call__(self, policies: List[PolicyInterface], times=1) -> List[Tuple[float, PolicyInterface]]:
        with Pool(cpu_count()) as p:
            # List[Tuple[float, PolicyInterface]]
            results = p.starmap(self._eval_policy, policies)
        return results

    def _eval_policy(self, policy: PolicyInterface, times=1):
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

