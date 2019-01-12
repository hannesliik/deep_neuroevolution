import time
from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, Optional

import numpy as np
import torch

from api.evaluators import Evaluator, Score
from api.utils import Policy


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
        self._elites: List[Policy] = []

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
        if self._elites:
            elites_checked = self.evaluator([elite[0] for elite in self._elites], 20)
            best_policy = sorted(elites_checked, key=lambda x: float(x[1]), reverse=True)[0]
            self.state["frames_evaluated"] += sum([elite[1]["n_frames"] for elite in self._elites])
            return best_policy[0]

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
        self._elites = sorted(prev_gen, key=lambda pol_sc: float(pol_sc[1]), reverse=True)[:self.n_elites]
        return self._elites

    def _pick_parents(self, elites: List[Tuple[Policy, Score]], n=2, p=None) -> List[Policy]:
        return [elites[i][0] for i in
                np.random.choice(len(elites), size=n, replace=False, p=p)]

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
                 parent_selection: str = "uniform", std: float = 0.02,
                 size: int = 200, n_elites: int = 20,
                 decay: float = 1.):
        """
        :param parent_selection: "uniform" to choose parents from elites uniformly,
        "probab" - probabilistically based on rewards
        :param std: Standard deviation of the Gaussian mutation
        :param decay: std decay - every iteration the std is multiplied by the value. The "annealing" parameter.
        """
        params = {"decay": decay, "strategy": "GaussianMutationStrategy", "parent_selection": parent_selection,
                  "std": std, "size": size,
                  "n_elites": n_elites}
        super().__init__(policy_factory, evaluator, size=size, n_elites=n_elites, params=params)
        self.parent_selection = parent_selection
        self.std = std
        self.decay = decay

    def __call__(self, *args, **kwargs):
        generation = super().__call__(*args, **kwargs)
        self.std *= self.decay
        return generation

    def _generate(self, elites: List[Tuple[Policy, Score]]) -> List[Policy]:
        offspring = [e[0] for e in elites]
        p = softmax([float(elite[1]) for elite in elites]) if self.parent_selection == "probab" else None
        for i in range(self.size - self.n_elites):
            parent = self._pick_parents(elites, n=1, p=p)[0]
            child = self._generate_child(parent)
            offspring.append(child)
        return offspring

    def _generate_child(self, parent: Policy):
        child = self.policy_factory()
        child.load_state_dict(parent.state_dict())
        for tensor in child.state_dict().values():
            tensor += torch.randn_like(tensor) * self.std
        return child


class CrossoverStrategy(AbsEvoStrategy):
    """
    Uses parts of parents pairs to generate offspring
    """

    def __init__(self, policy_factory: Callable, evaluator: Evaluator,
                 parent_selection: str = "uniform",
                 std: float = 0.1, p_crossover: float = 0.3,
                 size: int = 200, n_elites: int = 20,
                 decay: float = 1.):
        """
        :param parent_selection: "uniform" to choose parents from elites uniformly,
        "probab" - probabilistically based on rewards
        """
        params = {"strategy": "CrossoverStrategy", "parent_selection": parent_selection,
                  "size": size, "n_elites": n_elites, "std": std, "p_crossover": p_crossover,
                  "decay": decay}
        super().__init__(policy_factory, evaluator, size=size, n_elites=n_elites, params=params)
        self.parent_selection = parent_selection
        self.p_crossover = p_crossover
        self.std = std
        self.decay = decay

    def __call__(self, *args, **kwargs):
        generation = super().__call__(*args, **kwargs)
        self.std *= self.decay
        self.p_crossover *= self.decay
        return generation

    def _generate(self, elites: List[Tuple[Policy, Score]]) -> List[Policy]:
        offspring = [e[0] for e in elites]
        p = softmax([float(elite[1]) for elite in elites]) if self.parent_selection == "probab" else None
        for i in range(self.size - self.n_elites):
            parents = self._pick_parents(elites, n=2, p=p)
            children = self._generate_children(parents[0], parents[1])
            offspring += children
        return offspring

    def _generate_children(self, parent1: Policy, parent2: Policy):
        child1, child2 = self.policy_factory(), self.policy_factory()
        child1.load_state_dict(parent1.state_dict())
        child2.load_state_dict(parent2.state_dict())
        for tensor1, tensor2 in zip(child1.state_dict().values(), child2.state_dict().values()):
            if np.random.rand() > self.p_crossover:
                temp = tensor1.clone()
                tensor1.copy_(tensor2)
                tensor2.copy_(temp)
            tensor1 += torch.randn_like(tensor1) * self.std
            tensor2 += torch.randn_like(tensor2) * self.std
        return child1, child2

'''
class CovMatAdaptationStrategy(AbsEvoStrategy):
    """
    Modifies elite members by applying constant Gaussian mutation
    """

    def __init__(self, policy_factory: Callable, evaluator: Evaluator,
                 size: int = 200, n_elites: int = 20,
                 n_check_top: int = 10, n_check_times: int = 30):
        params = {"strategy": "CovMatAdaptationStrategy", "size": size,
                  "n_elites": n_elites, "n_check_top": n_check_top, "n_check_times": n_check_times}
        super().__init__(policy_factory, evaluator, size=size, n_elites=n_elites,
                         n_check_top=n_check_top, n_check_times=n_check_times,
                         params=params)
        self._means = None

    def __call__(self, prev_gen: List[Policy], gen_number: Optional[int] = None):
        if self._means is None:
            self._means = self._calc_means(self._collect_vars(prev_gen))
        return super().__call__(prev_gen, gen_number)

    def _generate(self, elites: List[Tuple[Policy, Score]]) -> List[Policy]:
        vars = self._collect_vars([elite[0] for elite in elites])
        cov = self._calc_cov(vars, self._means)
        self._means = self._calc_means(vars)
        sample = np.random.multivariate_normal(self._means, cov, size=self.size)
        offspring = []
        for state in sample:
            child = self.policy_factory()
            i = 0
            for tensor in child.state_dict().values():
                data = torch.tensor(state[i:i + tensor.numel()]).reshape(tensor.shape)
                tensor[:] = data
                i += tensor.numel()
            offspring.append(child)

        true_elite = self._find_true_elite(elites)
        self._best_policy = true_elite
        offspring.append(true_elite)
        return offspring

    def _collect_vars(self, policies: List[Policy]) -> np.ndarray:
        """
        :return: [n_vars x n_policies]
        """
        return np.stack([np.concatenate([tensor.numpy().flatten()
                                         for tensor in policy.state_dict().values()])
                         for policy in policies]).transpose()

    def _calc_means(self, vars: np.ndarray) -> np.ndarray:
        """
         :return: [n_vars x 1]
         """
        return np.mean(vars, axis=1)

    def _calc_cov(self, vars: np.ndarray, means=None) -> np.ndarray:
        """
         :return: [n_vars x n_vars]
         """
        if means is None:
            means = self._calc_means(vars)
        vars -= np.expand_dims(means, axis=1)
        cov = np.dot(vars, vars.transpose())
        cov *= np.true_divide(1, vars.shape[1] - 1)
        return cov.squeeze()

'''
# TODO implement more strategies
