from abc import ABC, abstractmethod
from typing import List, Tuple, Callable

import numpy as np
import torch
from scipy.special import softmax

from api.evaluators import Evaluator, Score
from api.utils import Policy


# TODO Improve hierarchy


class EvoStrategy(ABC):
    """
    Evolution Strategy interface.
    The strategy should accept a list of policies, evaluate them, perform mutations,
    and produce the next generation
    """

    @abstractmethod
    def __call__(self, prev_gen: List[Policy]) -> List[Policy]:
        """
        Produces next generation of policies
        :param prev_gen: previous generation of policies
        :return: xext generation of policies
        """
        pass


class AbsEvoStrategy(EvoStrategy):
    """
    Abstract implementation of the EvoStrategy interface with basic common pipeline
    """

    @abstractmethod
    def __init__(self, policy_factory: Callable, evaluator: Evaluator, size: int = 200,
                 n_elites: int = 20):
        """
        :param policy_factory: factory for creating policies
        :param evaluator: evaluator used for estimating policies characteristics
        :param size: number of policies to generate
        :param n_elites: number of policies to consider when creating the next generation
        """
        self.policy_factory = policy_factory
        self.evaluator = evaluator
        self.size = size
        self.n_elites = n_elites

    def __call__(self, prev_gen: List[Policy]) -> List[Policy]:
        """
        :param prev_gen: previous generation of policies
        :return: xext generation of policies
        """
        evaluated = self._evaluate(prev_gen)
        elites = self._select_elites(evaluated)
        offspring = self._generate(elites)
        return offspring

    def _evaluate(self, prev_gen: List[Policy]) -> List[Tuple[Policy, Score]]:
        """
        Evaluate scores of the policies

        :param prev_gen: generation of policies to be evaluated
        :return: pairs of policies and their evaluated scores
        """
        return self.evaluator(prev_gen)

    def _select_elites(self, prev_gen: List[Tuple[Policy, Score]]) -> List[Tuple[Policy, Score]]:
        """
        Select subset of the previous generation as eligible to be used in producing the next generation

        :param prev_gen: previous generation of policies and their scores
        :return: list of elites and their scores
        """
        return sorted(prev_gen, key=lambda pol_sc: float(pol_sc[1]), reverse=True)[:self.n_elites]

    @abstractmethod
    def _generate(self, elites: List[Tuple[Policy, Score]]) -> List[Policy]:
        """
        Generate next generation based on eligible elites

        :param elites: list of elites and their scores
        :return: next generation of policies
        """
        pass


class GaussianMutationStrategy(AbsEvoStrategy):
    """
    Modifies elite members by applying constant Gaussian mutation
    """

    def __init__(self, policy_factory: Callable, evaluator: Evaluator,
                 parent_selection: str = "uniform", std=0.02,
                 size: int = 200, n_elites: int = 20,
                 n_check_top: int = 10, n_check_times: int = 30):
        """
        :param parent_selection: "uniform" to choose parents from elites uniformly,
        "probab" - probabilistically based on rewards
        :param std: Standard deviation of the Gaussian mutation
        """
        super().__init__(policy_factory, evaluator, size, n_elites)
        self.parent_selection = parent_selection
        self.std = std
        self.n_check_top = n_check_top
        self.n_check_times = n_check_times

    def _generate(self, elites: List[Tuple[Policy, Score]]) -> List[Policy]:
        offspring = []
        self.p = None
        for i in range(self.size):
            parent = self._pick_parent(elites)
            child = self._generate_child(parent)
            offspring.append(child)
        self.p = None
        offspring.append(self._find_true_elite(elites))
        return offspring

    def _pick_parent(self, elites: List[Tuple[Policy, Score]]) -> Policy:
        parent = None
        if self.parent_selection == "uniform":
            parent = elites[np.random.randint(len(elites))][0]
        elif self.parent_selection == "probab":
            if self.p is None:
                self.p = softmax([float(elite[1]) for elite in elites])
            parent = elites[np.random.choice(len(elites), p=self.p)][0]
        return parent

    def _find_true_elite(self, elites: List[Tuple[Policy, Score]]) -> Policy:
        elites_checked = self.evaluator([elite[0] for elite in elites[:self.n_check_top]], self.n_check_times)
        true_elite = sorted(elites_checked, key=lambda x: float(x[1]), reverse=True)[0]
        print(true_elite[1])
        return true_elite[0]

    def _generate_child(self, parent: Policy):
        child = self.policy_factory()
        child.load_state_dict(parent.state_dict())
        for tensor in child.state_dict().values():
            tensor += torch.randn_like(tensor) * self.std
        return child


class NoveltySearchStrategy(GaussianMutationStrategy):
    """
    Produces next generation based on the novelty estimated by an arbitrary metric
    """

    def __init__(self, policy_factory: Callable, evaluator: Evaluator,
                 novelty_metric: Callable, n_neighbors = 15,
                 parent_selection: str = "uniform", std=0.02,
                 size: int = 200, n_elites: int = 20,
                 n_check_top: int = 10, n_check_times: int = 30):
        """
        :param nov_dist: Tuple[Tuple[Policy, Score], Tuple[Policy, Score]] -> float
        - distance metric of novelty between policies
        :param n_neighbors: number of neighbors to consider calculating the overall novelty
        """
        super().__init__(policy_factory, evaluator, parent_selection, std, size, n_elites, n_check_top, n_check_times)
        self.nov_dist = novelty_metric
        self.n_neighbors = n_neighbors
        self.archive = []

    def _select_elites(self, prev_gen: List[Tuple[Policy, Score]]) -> List[Tuple[Policy, Score]]:
        neighbors = self.archive + prev_gen
        # consider more efficient computation method
        novelty_scores: List[Tuple[Tuple[Policy, Score], float]] = []
        for i in neighbors:
            dists_i = []
            for j in neighbors:
                dists_i.append(self.nov_dist(i, j))
            novelty = np.mean(dists_i.sort(reverse=True)[:self.n_neighbors])
            novelty_scores.append((i, novelty))
        return [policy for policy, novelty
                in novelty_scores.sort(key=lambda x: x[1])[:self.n_elites]]

# TODO implement more strategies
