import numpy as np
import torch
import gym
import time

import os
#os.environ["OMP_NUM_THREADS"] = "1"
from evolution_strategies import BasicStrategy
from evaluators import ParallelEnvEvaluator
from deep_neuroevolution import GAOptimizer
from utils import Policy

# Disable annoying warnings from gym
gym.logger.set_level(40)


# Define policy
class HCTorchPolicy(Policy, torch.nn.Module):
    # Input and output sizes from the HalfCheetah environment
    N_INPUTS = 17  # env.action_space.shape
    N_OUTPUTS = 6  # env.observation_space.shape

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(HCTorchPolicy.N_INPUTS, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, HCTorchPolicy.N_OUTPUTS))

    def forward(self, x):
        return self.net.forward(x)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        # Observation to torch tensor, add empty batch dimension
        obs = torch.from_numpy(obs).float().unsqueeze(0)
        action = self.net.forward(obs)
        return action.detach().numpy()  # Back to numpy array and return


# Create environment factory
def env_factory() -> gym.Env:
    return gym.make("HalfCheetah-v2")


# Create policy factory
def policy_factory() -> Policy:
    policy = HCTorchPolicy()
    # Tell torch that we will not calculate gradients.
    # Up to ~10% speedup and maybe takes less memory
    for param in policy.parameters():
        param.requires_grad = False
    return policy


if __name__ == '__main__':

    # Create evaluator
    evaluator = ParallelEnvEvaluator(env_factory=env_factory, times=2, n_processes=16)
    evolution_strategy = BasicStrategy(evaluator, policy_factory, generation_size=1000, n_elites=20, n_check_top=10,
                                       n_check_times=30, std=0.02)


    def eval_callback(results):
        results, best_policy, info_dict = results
        rewards = [result[0] for result in results]
        print(np.mean(rewards), rewards[:10], info_dict["n_frames"])


    # eval_callback = lambda results: print(np.mean(results), results[:10])
    optimizer = GAOptimizer(env_factory, policy_factory, evolution_strategy, evaluator, eval_callback=eval_callback)
    for _ in range(20):
        start = time.time()
        optimizer.train_generation()
        print("generation time:", time.time() - start)
