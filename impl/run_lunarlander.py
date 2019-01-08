import json
import os
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from api.deep_neuroevolution import GAOptimizer
from api.evaluators import ParallelEnvEvaluator
from api.evolution_strategies import GaussianMutationStrategy, CrossoverStrategy
from api.utils import Policy, ObsNormalizer

# Disable annoying warnings from gym
gym.logger.set_level(40)


# Define policy
class LunarLanderTorchPolicy(Policy, torch.nn.Module):
    N_INPUTS = 8  # env.action_space.shape
    N_OUTPUTS = 4  # env.observation_space.shape

    def __init__(self, obs_normalizer: ObsNormalizer):
        super().__init__()
        self.obs_normalizer = obs_normalizer
        self.net = torch.nn.Sequential(
            torch.nn.Linear(LunarLanderTorchPolicy.N_INPUTS, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, LunarLanderTorchPolicy.N_OUTPUTS),
            torch.nn.Softmax(dim=1))

    def forward(self, x):
        return self.net.forward(x)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        # Observation to torch tensor, add empty batch dimension
        obs = self.obs_normalizer.normalize(obs)
        obs = torch.from_numpy(obs).float().unsqueeze(0)
        action = self.net.forward(obs)
        return torch.argmax(action, dim=1).detach().numpy()[0]  # Back to numpy array and return


# Create environment factory
def env_factory() -> gym.Env:
    return gym.make("LunarLander-v2")


obs_normalizer = ObsNormalizer(env_factory, n_samples=2000)


# Create policy factory
def policy_factory() -> Policy:
    policy = LunarLanderTorchPolicy(obs_normalizer)
    # Tell torch that we will not calculate gradients.
    # Up to ~10% speedup and maybe takes less memory
    for param in policy.parameters():
        param.requires_grad = False
    return policy


if __name__ == '__main__':

    # Create evaluator
    evaluator = ParallelEnvEvaluator(env_factory=env_factory, times=3)

    env = env_factory()

    '''
    evolution_strategy = GaussianMutationStrategy(policy_factory, evaluator=evaluator,
                                                  parent_selection="uniform",
                                                  std=0.1,
                                                  size=1000, n_elites=20, n_check_top=10, n_check_times=30,
                                                  decay=0.97)
    '''
    evolution_strategy = CrossoverStrategy(policy_factory, evaluator=evaluator,
                                            parent_selection="uniform",
                                            size=1000, n_elites=20)

    optimizer = GAOptimizer(env_factory, policy_factory, evolution_strategy, evaluator)

    experiment_name = time.strftime("%Y%m%d_%H%M%S") + "_lunar_lander"
    if not os.path.exists("data"):
        os.makedirs("data/")
    if not os.path.exists("data/" + experiment_name):
        os.makedirs("data/" + experiment_name)
    prefix = "data/" + experiment_name + "/"

    with open(prefix + "params.json", "w") as fp:
        json.dump(evolution_strategy.state["params"], fp)

    for i in range(50):
        start = time.time()
        optimizer.train_generation()
        # print(evolution_strategy.state)
        print("generation time:", time.time() - start)
        # Save model
        torch.save(evolution_strategy.best_policy.state_dict(), prefix + "model_state_dict.pth")
        # Generate plot
        data = evolution_strategy.state["evaluations"]
        df = pd.DataFrame(data)
        sns.lineplot(data=df, x="time", y="score", ci="sd")
        plt.savefig(prefix + f"plot_{i}.png")
        df.to_csv(prefix + "plot.csv")
