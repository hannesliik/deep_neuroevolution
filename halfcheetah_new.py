import numpy as np
import torch
import gym
from evolution_strategies import BasicStrategy
from evaluators import ParallelEnvEvaluator
from deep_neuroevolution import GAOptimizer
from utils import Policy


# Define policy
class HCTorchPolicy(Policy, torch.nn.Module):
    # Input and output sizes from the HalfCheetah environment
    N_INPUTS = 17
    N_OUTPUTS = 6

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(HCTorchPolicy.N_INPUTS, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, HCTorchPolicy.N_OUTPUTS))

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
    return HCTorchPolicy()

if __name__ == '__main__':
    gym.logger.set_level(40)
    # Create evaluator
    evaluator = ParallelEnvEvaluator(env_factory=env_factory)
    evolution_strategy = BasicStrategy(evaluator, policy_factory(), generation_size=1000, n_elites=20, n_check_top=10,
                                       n_check_times=30)
    optimizer = GAOptimizer(env_factory, policy_factory, evolution_strategy, evaluator)
