import gym
from typing import Tuple, List, Callable
from multiprocessing import Pool, cpu_count

import numpy as np

from utils import Policy, ObsNormalizer
from evolution_strategies import EvolutionaryStrategy
from evaluators import Evaluator, ParallelEnvEvaluator

class GAOptimizer:

    def __init__(self, env_factory: Callable,
                 policy_factory: Callable,
                 evolution_strategy: EvolutionaryStrategy,
                 evaluator: Evaluator):
        self._env_factory = env_factory
        self._model_factory = policy_factory
        self.evaluator = evaluator
        self.evolution_strategy: EvolutionaryStrategy = evolution_strategy
        self.normalizer: ObsNormalizer = ObsNormalizer(env_factory, n_samples=1000)
        self.best_policy = policy_factory()
        self.generation: List[Policy] = [self.best_policy]
        # something_callback: function


    def train_generation(self):
        with Pool(cpu_count()) as p:
            # List[Tuple[float, PolicyInterface]]
            eval_results, self.best_policy = self.evaluator(self.generation)
        self.generation = self.evolution_strategy(eval_results)

