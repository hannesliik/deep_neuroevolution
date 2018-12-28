from abc import ABC, abstractmethod
from typing import List, Tuple, Callable
from multiprocessing import Pool, cpu_count

from gym import Env
import numpy as np

from utils import Policy


class Evaluator(ABC):
    """
    Evaluator, when called, should take in a list of policies and return a
    list of tuples (score, policy) and the best policy and any additional information that an implementation provides
    """
    @abstractmethod
    def __call__(self, policies: List[Policy]) -> Tuple[List[Tuple[float, Policy]], Policy, dict]:
        pass


class SequentialEnvEvaluator(Evaluator):
    """
    Evaluator that takes environments that implement gym.Env. Makes only one environment and iterates through
    all the policies sequentially through that environment.
    """
    def __init__(self, env_factory: Callable):
        self.env: Env = env_factory()

    def __call__(self, policies: List[Policy], times=1) -> Tuple[List[Tuple[float, Policy]], Policy, dict]:
        results: List[Tuple[float, Policy]] = []
        n_frames = 0
        for policy in policies:
            rewards = []
            policy.set_up()
            for _ in range(times):
                obs = self.env.reset()
                done = False
                total_reward = 0
                while not done:
                    n_frames += 1
                    action = policy(obs)
                    obs, reward, done = self.env.step(action)[:3]
                    total_reward += reward
                rewards.append(total_reward)
            policy.teardown()
            results.append((int(np.mean(rewards)), policy))
        results = sorted(results, key=lambda x: x[0], reverse=True)
        return results, results[0][1], {"n_frames": n_frames}


class ParallelEnvEvaluator(Evaluator):
    """
    Gym environment evaluator. Uses a process pool to evaluate policies. A new environment instance is created
    for each policy.
    """
    def __init__(self, env_factory: Callable, times=1, n_processes: int = cpu_count()):
        self.env_factory = env_factory
        self.n_processes = n_processes
        self.times = times

    def __call__(self, policies: List[Policy], times=None) -> Tuple[List[Tuple[float, Policy]], Policy, dict]:
        if times is None:
            times = self.times
        with Pool(self.n_processes) as p:
            # List[Tuple[float, PolicyInterface]]
            results = p.starmap(self._eval_policy, [(policy, times) for policy in policies])
        results = sorted(results, key=lambda x: x[0], reverse=True)
        n_frames = sum([result[2] for result in results])
        return results, results[0][1], {"n_frames": n_frames}

    def _eval_policy(self, policy: Policy, times=1) -> Tuple[float, Policy, int]:
        """
        Function to evaluate one policy
        :param policy: some function that produces actions for given observations
        :param times: How many times the policy will be evaluated. Returns the mean reward across runs
        :return: (float- mean reward across runs, policy - the policy instance, int - number of observations seen)
        """
        env: Env = self.env_factory()
        rewards = []
        policy.set_up()
        n_frames = 0
        for _ in range(times):
            obs = env.reset()
            done = False
            total_reward = 0
            while not done:
                n_frames += 1
                action = policy(obs)
                obs, reward, done = env.step(action)[:3]
                total_reward += reward
            rewards.append(total_reward)
        env.close()
        policy.teardown()
        return float(np.mean(rewards)), policy, n_frames
