from abc import ABC, abstractmethod
from typing import List, Tuple, Callable
import random

import torch

from utils import Policy
from evaluators import Evaluator


class EvolutionaryStrategy(ABC):
    """
    Evolution Strategy Interface.
    The strategy should take a list of policies and their rewards, do some mutations
    or combinations or what not and return a list of new policies as the next generation
    """

    @abstractmethod
    def __call__(self, prev_gen: List[Tuple[float, Policy]]) -> List[Policy]:
        """
        Input is a list of tuples (reward, policy)
        Output is the new generation of policies
        :param prev_gen: Evaluations of previous generation
        :return: next generation
        """
        pass


class BasicStrategy(EvolutionaryStrategy):
    """
    The standard evolution strategy. Presumes pytorch policies
    """

    def __init__(self, evaluator: Evaluator, policy_factory: Callable, generation_size: int = 1000,
                 n_elites: int = 20,
                 n_check_top: int = 10, n_check_times: int = 30, std=0.02):
        self.n_elites = n_elites
        self.gen_size = generation_size
        self.eval_fn = evaluator
        self.policy_factory = policy_factory
        # Params for finding the true elite
        self.n_check_top = n_check_top
        self.n_check_times = n_check_times
        self.std = std

    def _gen_population(self, elites, n_models):
        offspring = []
        for i in range(n_models):
            parent = elites[random.randint(0, len(elites) - 1)]
            parent_state_dict = parent.state_dict()
            policy = self.policy_factory()
            policy.load_state_dict(parent_state_dict)
            for tensor in policy.state_dict().values():
                tensor += torch.randn_like(tensor) * self.std
            offspring.append(policy)
        return offspring

    def __call__(self, prev_gen: List[Tuple[int, Policy]]) -> List[Policy]:
        # Sort the policies by rewards and take top n_elites
        elites = sorted(prev_gen, key=lambda x: x[0], reverse=True)[:self.n_elites]

        # Throw away the rewards to get a list of policies
        elites = [elite[1] for elite in elites]

        # Evaluate the top n_check_top elites to
        elites_checked, _ = self.eval_fn(elites[:self.n_check_top], self.n_check_times)
        elites_checked = sorted(elites_checked, key=lambda x: x[0], reverse=True)

        offspring = self._gen_population(elites, self.gen_size)
        offspring.append(elites_checked[0][1])
        return offspring
