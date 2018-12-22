from abc import ABC, abstractmethod
from typing import List, Tuple
from multiprocessing import Pool

import torch


class ESInterface(ABC):
    """
    Evolution Strategy Interface.
    The strategy should take a list of policies and their rewards, do some mutations
    or combinations or what not and return a list of new policies as the next generation
    """

    @abstractmethod
    def gen_population(self, prev_gen: List[Tuple[int, PolicyInterface]]) -> List[PolicyInterface]:
        """
        Input is a list of tuples (reward, policy)
        Output is the new generation of policies
        :param prev_gen: Evaluations of previous generation
        :return: next generation
        """
        pass


class BasicStrategy(ESInterface):
    """
    The standard evolution strategy. Presumes pytorch policies
    """

    def __init__(self, eval_function: function, policy_factory: function, generation_size: int = 1000,
                 n_elites: int = 20,
                 n_check_top: int = 10, n_check_times: int = 30):
        self.n_elites = n_elites
        self.gen_size = generation_size
        self.eval_fn = eval_function
        self.policy_factory = policy_factory
        # Params for finding the true elite
        self.n_check_top = n_check_top
        self.n_check_times = n_check_times

    def _gen_population(self, elites, n_models):
        offspring = []
        for i in range(n_models):
            parent = elites[random.randint(0, len(elites) - 1)]
            parent_state_dict = parent.state_dict()
            policy = self.policy_factory()
            policy.load_state_dict(parent_state_dict)
            for tensor in policy.state_dict().values():
                mutation = torch.randn_like(tensor) * 0.02
                tensor += mutation
            offspring.append(policy)
        return offspring

    def gen_population(self, prev_gen: List[Tuple[int, PolicyInterface]]) -> List[PolicyInterface]:
        # Sort the policies by rewards and take top n_elites
        elites = sorted(prev_gen, key=lambda x: x[0], reverse=True)[:self.n_elites]

        # Throw away the rewards to get a list of policies
        elites = [elite[1] for elite in elites]

        with Pool(cpu_count()) as p:
            # Evaluate the top n_check_top elites to
            elites_checked = p.starmap(self.eval_fn, [(e, self.n_check_times) for e in elites[:self.n_check_top]])
            elites_checked = sorted(elites_checked, key=lambda x: x[0], reverse=True)

        offspring = self._gen_population(elites, self.gen_size)
        offspring.append(elites_checked[0][1])
        return offspring
