
from typing import List, Callable

from api.evolution_strategies import EvoStrategy
from api.utils import Policy


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
        self.generation: List[Policy] = [policy_factory() for i in range(evolution_strategy.size)]

    def train_generation(self):
        # List[Tuple[float, PolicyInterface]]
        self.generation = self.evolution_strategy(self.generation)

