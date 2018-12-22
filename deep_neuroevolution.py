import gym
from typing import Tuple, List
from multiprocessing import Pool, cpu_count

import numpy as np

from utils import Policy, ObsNormalizer
from evolution_strategies import EvolutionaryStrategy
from evaluators import Evaluator, ParallelEnvEvaluator

class GAOptimizer:

    def __init__(self, env_factory: function,
                 policy_factory: function,
                 evolution_strategy: EvolutionaryStrategy,
                 evaluator: Evaluator):
        self._env_factory: function = env_factory
        self._model_factory: function = policy_factory
        self.evaluator = evaluator
        self.evolution_strategy: EvolutionaryStrategy = evolution_strategy
        self.normalizer: ObsNormalizer = ObsNormalizer(env_factory, n_samples=1000)
        self.generation: List[Policy] = [policy_factory()]
        # something_callback: function


    def train_generation(self):
        with Pool(cpu_count()) as p:
            # List[Tuple[float, PolicyInterface]]
            eval_results = self.evaluator(self.generation)
        self.generation = self.evolution_strategy(eval_results)

