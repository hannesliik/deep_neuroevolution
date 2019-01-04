
from typing import List, Callable

from api.utils import Policy
from api.evolution_strategies import EvoStrategy


class GAOptimizer:

    def __init__(self, env_factory: Callable,
                 policy_factory: Callable,
                 evolution_strategy: EvoStrategy,
                 eval_callback: Callable = None):
        self.eval_callback = eval_callback
        self._env_factory = env_factory
        self._model_factory = policy_factory
        self.evolution_strategy: EvoStrategy = evolution_strategy
        self.best_policy = policy_factory()
        self.generation: List[Policy] = [self.best_policy]

    def train_generation(self):
        # List[Tuple[float, PolicyInterface]]
        self.generation = self.evolution_strategy(self.generation)

