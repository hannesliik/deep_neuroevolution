from abc import ABC, abstractmethod
from typing import List, Tuple, Callable

import numpy as np
import torch
from scipy.special import softmax

from api.evaluators import Evaluator, Score
from api.utils import Policy


class EvoStrategy(ABC):
    """
    Evolution Strategy interface.
    The strategy should take a list of policies and their rewards, do some mutations
    or combinations or what not and return a list of new policies as the next generation
    """

    @abstractmethod
    def __init__(self, policy_factory: Callable, size: int = 200):
        """
        :param policy_factory: Factory for creating policies
        :param size: Number of policies to generate
        """
        self.policy_factory = policy_factory
        self.size = size


    @abstractmethod
    def __call__(self, prev_gen: List[Tuple[Policy, Score]]) -> List[Policy]:
        """
        Input is a list of tuples (reward, policy)
        Output is the new generation of policies
        :param prev_gen: Evaluations of previous generation
        :return: next generation
        """
        pass


class GaussianMutationStrategy(EvoStrategy):
    """
    Modifies elite members by applying constant Gaussian mutation
    """


    """
    param parent_selection: options: "uniform", "probab"
    """
    def __init__(self, policy_factory: Callable, size: int = 200,
                 n_elites: int = 20, elite_selection="score",
                 parent_selection="uniform", std=0.02,
                 evaluator: Evaluator = None, n_check_top: int = 10, n_check_times: int = 30):
        """
        :param n_elites: number of elites to select
        :param elite_selection: "score" to choose elites by score,
        "novelty" to perform novelty search
        :param parent_selection: "uniform" to choose parents from elites uniformly,
        "probab" - probabilistically based on rewards
        :param std: Standard deviation of the Gaussian mutation
        :param evaluator: Evaluator to check for true elite
        :param n_check_top: Number of elites to consider as true elite
        :param n_check_times: Number of times to check true elite reward
        """
        super().__init__(policy_factory, size)
        self.elite_selection = elite_selection
        if elite_selection == "novelty":
            self.policy_archive = []
        self.n_elites = n_elites
        self.parent_selection = parent_selection
        self.std = std
        self.evaluator = evaluator
        self.n_check_top = n_check_top
        self.n_check_times = n_check_times

    def __call__(self, prev_gen: List[Tuple[Policy, Score]]) -> List[Policy]:
        elites = self._select_elites(prev_gen)
        offspring = self._gen_population(elites, self.size)

        if self.evaluator:
            # Check for the true elite and retain it within population
            elites_checked = self.evaluator([elite[0] for elite in elites[:self.n_check_top]], self.n_check_times)
            elites_checked = sorted(elites_checked, key=lambda x: float(x[1]), reverse=True)
            offspring.append(elites_checked[0][0])

        return offspring

    def _select_elites(self, prev_gen: List[Tuple[Policy, Score]]):
        if self.elite_selection == "score":
            return sorted(prev_gen, key=lambda x: float(x[1]), reverse=True)[:self.n_elites]
        elif self.elite_selection == "novelty":
            pass #TODO

    def _gen_population(self, elites, size):
        offspring = []
        elites_p = None
        for i in range(size):
            if self.parent_selection == "uniform":
                parent = elites[np.random.randint(len(elites))][0]
            elif self.parent_selection == "probab":
                if elites_p is None:
                    elites_p = softmax([float(elite[1]) for elite in elites])
                parent = elites[np.random.choice(len(elites), p=elites_p)][0]

            policy = self.policy_factory()
            policy.load_state_dict(parent.state_dict())
            for tensor in policy.state_dict().values():
                tensor += torch.randn_like(tensor) * self.std

            offspring.append(policy)
        return offspring

