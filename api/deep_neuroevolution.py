
from typing import List, Callable

from api.evolution_strategies import EvoStrategy
from api.utils import Policy


class GAOptimizer:

    def __init__(self,
                 policy_factory: Callable,
                 evolution_strategy: EvoStrategy):
        self.evolution_strategy: EvoStrategy = evolution_strategy
        self.generation: List[Policy] = [policy_factory() for i in range(evolution_strategy.size)]
        self.iteration = 0

    def train_generation(self):
        # List[Tuple[float, PolicyInterface]]
        self.generation = self.evolution_strategy(self.generation, gen_number=self.iteration)
        self.iteration += 1

