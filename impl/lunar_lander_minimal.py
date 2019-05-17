import time

import gym
import numpy as np
import torch

from api.deep_neuroevolution import GAOptimizer
from api.evaluators import ParallelEnvEvaluator
from api.evolution_strategies import GaussianMutationStrategy
from api.utils import Policy

# Disable annoying warnings from gym
gym.logger.set_level(40)


# Define policy
class LunarLanderTorchPolicy(Policy, torch.nn.Module):
    N_INPUTS = 8  # env.action_space.shape
    N_OUTPUTS = 4  # env.observation_space.shape

    def __init__(self):
        super().__init__()
        # self.obs_normalizer = obs_normalizer
        self.net = torch.nn.Sequential(
            torch.nn.Linear(LunarLanderTorchPolicy.N_INPUTS, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, LunarLanderTorchPolicy.N_OUTPUTS),
            torch.nn.Softmax(dim=1))

    def forward(self, x):
        return self.net.forward(x)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        # Observation to torch tensor, add empty batch dimension
        obs = torch.from_numpy(obs).float().unsqueeze(0)
        action = self.net.forward(obs)
        return torch.argmax(action, dim=1).detach().numpy()[0]  # Back to numpy array and return


# Create environment factory
def env_factory() -> gym.Env:
    return gym.make("LunarLander-v2")


# Create policy factory
def policy_factory() -> Policy:
    policy = LunarLanderTorchPolicy()
    # Tell torch that we will not calculate gradients.
    # Up to ~10% speedup and maybe takes less memory
    for param in policy.parameters():
        param.requires_grad = False
    return policy


if __name__ == '__main__':

    args = {
        "gen_size": 200,  # Generation size
        "elites": 20, # Number of elites
        "check_times": 1,
        "check_n": 1,
        "decay": 0.97,
        "iterations": 25,
    }

    # Create evaluator
    evaluator = ParallelEnvEvaluator(env_factory)

    env = env_factory()
    evolution_strategy = GaussianMutationStrategy(policy_factory, evaluator=evaluator,
                                                  decay=args["decay"],
                                                  size=args["gen_size"], n_elites=args["elites"],
                                                  n_check_top=args["check_n"],
                                                  n_check_times=args["check_times"])

    optimizer = GAOptimizer(policy_factory, evolution_strategy)

    for i in range(args["iterations"]):
        print("Iteration", i)
        start = time.time()
        optimizer.train_generation()
        # print(evolution_strategy.state)
        print("generation time:", time.time() - start)
        results = evolution_strategy.state["evaluations"]
        latest = [result["score"] for result in results if result["generation"] == i]
        print(np.mean(latest))
        print(np.max(latest))
        # Save model
        torch.save(evolution_strategy.best_policy.state_dict(), "model_state_dict.pth")
