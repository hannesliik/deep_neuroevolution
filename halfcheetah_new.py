import numpy as np
import torch
import gym
import time

from evolution_strategies import BasicStrategy
from evaluators import ParallelEnvEvaluator
from deep_neuroevolution import GAOptimizer
from utils import Policy

gym.logger.set_level(40)
# Define policy
class HCTorchPolicy(Policy, torch.nn.Module):
    # Input and output sizes from the HalfCheetah environment
    N_INPUTS = 17
    N_OUTPUTS = 6

    def __init__(self):
        """
        Original:
        out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
        out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
        out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        """
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
    return HCTorchPolicy()

if __name__ == '__main__':

    # Create evaluator
    evaluator = ParallelEnvEvaluator(env_factory=env_factory, times=2, n_processes=8)
    evolution_strategy = BasicStrategy(evaluator, policy_factory, generation_size=1000, n_elites=20, n_check_top=10,
                                       n_check_times=30, std=0.02)
    def eval_callback(results):
        rewards = [result[0] for result in results]
        print(np.mean(rewards), rewards[:10])
    #eval_callback = lambda results: print(np.mean(results), results[:10])
    optimizer = GAOptimizer(env_factory, policy_factory, evolution_strategy, evaluator, eval_callback=eval_callback)
    for _ in range(20):
        start = time.time()
        optimizer.train_generation()
        print("generation time:", time.time() - start)