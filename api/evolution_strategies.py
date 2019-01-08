import time
from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, Optional

import numpy as np
import torch

from api.evaluators import Evaluator, Score
from api.utils import Policy


# from scipy.special import softmax


def softmax(x) -> np.ndarray:
    exp = np.exp(x)
    return exp / np.sum(exp)


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
        :return: next generation of policies
        """
        pass

    @property
    @abstractmethod
    def best_policy(self) -> Policy:
        """
        Returns the current best policy
        :return:
        """
        pass

    @property
    @abstractmethod
    def state(self) -> dict:
        """
        Extra information about the evolution history can be stored here
        :return:
        """
        pass


class AbsEvoStrategy(EvoStrategy):
    """
    Abstract implementation of the EvoStrategy interface with basic common pipeline
    """

    @abstractmethod
    def __init__(self, policy_factory: Callable, evaluator: Evaluator, size: int = 200,
                 n_elites: int = 20, params: dict = None):
        """
        :param policy_factory: factory for creating policies
        :param evaluator: evaluator used for estimating policies characteristics
        :param size: number of policies to generate
        :param n_elites: number of policies to consider when creating the next generation
        :param params: parameters of the experiment
        """
        self.exp_params = params
        self.policy_factory = policy_factory
        self.evaluator = evaluator
        self.size = size
        self.n_elites = n_elites
        self.start_time = -1
        self._state = {"params": params, "frames_evaluated": 0, "stats": [], "evaluations": []}
        self._best_policy: Policy = None

    def __call__(self, prev_gen: List[Policy], gen_number: Optional[int] = None) -> List[Policy]:
        """
        :param prev_gen: previous generation of policies
        :return: next generation of policies
        """
        if self.start_time == -1:
            self.start_time = time.time()
        evaluated = self._evaluate(prev_gen)
        self._update_stats(evaluated, gen_number)
        elites = self._select_elites(evaluated)
        offspring = self._generate(elites)
        self._best_policy = elites[0][0]
        return offspring

    @property
    def best_policy(self):
        return self._best_policy

    def _update_stats(self, evaluated: List[Tuple[Policy, Score]], gen_number: Optional[int]) -> None:
        """
        Takes a list of evaluations and adds the stats to the evolution strategy state
        :param evaluated:
        """
        scores = [float(evaluation[1]) for evaluation in evaluated]
        reward_mean = np.mean(scores)
        reward_std = np.std(scores)
        reward_min = np.min(scores)
        reward_max = np.max(scores)
        timestamp = time.time() - self.start_time
        for score in scores:
            self.state["evaluations"].append(
                {"generation": gen_number, "frames": self._state["frames_evaluated"], "time": timestamp, "score": score})
        self.state["stats"].append({"generation":gen_number, "frames": self._state["frames_evaluated"], "time": timestamp,
                                    "mean": reward_mean, "std": reward_std, "min": reward_min, "max": reward_max})
        # return

    def _evaluate(self, prev_gen: List[Policy]) -> List[Tuple[Policy, Score]]:
        """
        Evaluate scores of the policies

        :param prev_gen: generation of policies to be evaluated
        :return: pairs of policies and their evaluated scores
        """
        eval_results = self.evaluator(prev_gen)
        frames_evaluated = sum([result[1]["n_frames"] for result in eval_results])
        self._state["frames_evaluated"] += frames_evaluated
        return eval_results

    def _select_elites(self, prev_gen: List[Tuple[Policy, Score]]) -> List[Tuple[Policy, Score]]:
        """
        Select subset of the previous generation as eligible to be used in producing the next generation

        :param prev_gen: previous generation of policies and their scores
        :return: list of elites and their scores
        """
        return sorted(prev_gen, key=lambda pol_sc: float(pol_sc[1]), reverse=True)[:self.n_elites]

    @property
    def state(self):
        return self._state

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
                 n_check_top: int = 10, n_check_times: int = 30, decay: float = 1.):
        """
        :param parent_selection: "uniform" to choose parents from elites uniformly,
        "probab" - probabilistically based on rewards
        :param std: Standard deviation of the Gaussian mutation
        :param decay: std decay - every iteration the std is multiplied by the value. The "annealing" parameter.
        """
        params = {"decay": decay, "strategy": "GaussianMutationStrategy", "parent_selection": parent_selection,
                  "std": std, "size": size,
                  "n_elites": n_elites, "n_check_top": n_check_top, n_check_times: n_check_times}
        super().__init__(policy_factory, evaluator, size, n_elites, params=params)
        self.parent_selection = parent_selection
        self.std = std
        self.n_check_top = n_check_top
        self.n_check_times = n_check_times
        self.decay = decay

    def __call__(self, *args, **kwargs):
        generation = super().__call__(*args, **kwargs)
        self.std *= self.decay
        return generation

    def _generate(self, elites: List[Tuple[Policy, Score]]) -> List[Policy]:
        offspring = []
        self.p = None
        for i in range(self.size):
            parent = self._pick_parent(elites)
            child = self._generate_child(parent)
            offspring.append(child)
        self.p = None
        true_elite = self._find_true_elite(elites)
        self._best_policy = true_elite
        offspring.append(true_elite)
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
        self.state["frames_evaluated"] += sum([elite[1]["n_frames"] for elite in elites])
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
                 novelty_metric: Callable, n_neighbors=15,
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
