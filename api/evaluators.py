from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Callable
from multiprocessing import Pool, cpu_count

from gym import Env
import numpy as np

from api.utils import Policy


class Score:
    """
    Score of a policy as calculated by evaluator. Must be convertible to a simple float score,
    but can also contain additional evaluator-specific information
    """
    def __init__(self, score: float, **kwargs):
        self.score = score
        self.params = {k: v for k, v in kwargs.items()}

    def __getitem__(self, param):
        return self.params.get(param, None)

    def __setitem__(self, key, value):
        self.params[key] = value

    def __float__(self):
        return self.score

    def __str__(self):
        return str(self.score)


class Evaluator(ABC):
    """
    Evaluator, when called, should take in a list of policies and return a
    list of tuples (policy, score)
    """
    @abstractmethod
    def __call__(self, policies: List[Policy], times: int = 1) -> List[Tuple[Policy, Score]]:
        pass


class EnvEvaluator(Evaluator, ABC):
    """
    Evaluates policies against a gym.Env instance.
    """

    def _eval_policy(self, policy: Policy, env: Env, times: int = 1) -> Tuple[Policy, Score]:
        """
        Function to evaluate one policy
        :param policy: some function that produces actions for given observations
        :param env: Environment for evaluation
        :param times: How many times the policy will be evaluated. Returns the mean reward across runs
        :return: (policy - the policy instance, float- mean reward across runs, info - dictionary of auxiliary info)
        """
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
        score = Score(np.mean(rewards), n_frames=n_frames, last_obs=obs)
        return policy, score


class SequentialEnvEvaluator(EnvEvaluator):
    """
    Evaluator that takes environments that implement gym.Env. Makes only one environment and iterates through
    all the policies sequentially through that environment.
    """
    def __init__(self, env_factory: Callable):
        self.env: Env = env_factory()

    def __call__(self, policies: List[Policy], times: int = 1) -> List[Tuple[Policy, Score]]:
        return [self._eval_policy(policy, self.env, times) for policy in policies]


class ParallelEnvEvaluator(EnvEvaluator):
    """
    Gym environment evaluator. Uses a process pool to evaluate policies. A new environment instance is created
    for each policy.
    """
    def __init__(self, env_factory: Callable, times: int = 1, n_processes: int = cpu_count()):
        self.env_factory = env_factory
        self.n_processes = n_processes
        self.times = times

    def __call__(self, policies: List[Policy], times = None) -> List[Tuple[Policy, Score]]:
        if times is None:
            times = self.times
        with Pool(self.n_processes) as p:
            results = p.starmap(self._eval_policy, [(policy, self.env_factory(), times) for policy in policies])
        return results

