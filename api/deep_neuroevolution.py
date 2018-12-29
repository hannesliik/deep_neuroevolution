
from typing import List, Callable


from api.utils import Policy
from api.evolution_strategies import EvoStrategy
from api.evaluators import Evaluator


class GAOptimizer:

    def __init__(self, env_factory: Callable,
                 policy_factory: Callable,
                 evolution_strategy: EvoStrategy,
                 evaluator: Evaluator,
                 eval_callback: Callable = None):
        self.eval_callback = eval_callback
        self._env_factory = env_factory
        self._model_factory = policy_factory
        self.evaluator = evaluator
        self.evolution_strategy: EvoStrategy = evolution_strategy
        self.best_policy = policy_factory()
        self.generation: List[Policy] = [self.best_policy]
        # something_callback: function

    def train_generation(self):
        # List[Tuple[float, PolicyInterface]]
        eval_results = self.evaluator(self.generation)
        if self.eval_callback is not None:
            self.eval_callback(eval_results)
        self.generation = self.evolution_strategy(eval_results)
