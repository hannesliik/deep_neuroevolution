import time

import gym
import numpy as np
import torch

from api.deep_neuroevolution import GAOptimizer
from api.evaluators import ParallelEnvEvaluator
from api.evolution_strategies import GaussianMutationStrategy
from api.utils import Policy, plot_data, ObsNormalizer

# Disable annoying warnings from gym
gym.logger.set_level(40)


# Define policy
class LunarLanderTorchPolicy(Policy, torch.nn.Module):
    # Input and output sizes from the HalfCheetah environment
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

obs_normalizer = ObsNormalizer(env_factory)
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


    def eval_callback(results):
        results = sorted(results, key=lambda x: float(x[1]), reverse=True)
        top_results = results[0]
        print(top_results[1], np.mean([float(x[1]) for x in results]))
        obs = env.reset()
        done = False
        while not done:
            env.render()
            action = top_results[0](obs)
            obs, _, done = env.step(action)[:3]

    evolution_strategy = GaussianMutationStrategy(policy_factory, evaluator=evaluator,
                                                        parent_selection="uniform",
                                                         std=0.1,
                                                         size=1000, n_elites=20, n_check_top=10, n_check_times=30
                                                         )


optimizer = GAOptimizer(env_factory, policy_factory, evolution_strategy, evaluator)
for _ in range(50):
    start = time.time()
    optimizer.train_generation()
    #print(evolution_strategy.state)
    print("generation time:", time.time() - start)
    plot_data(evolution_strategy.state["evaluations"])

