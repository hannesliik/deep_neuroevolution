import numpy as np
import torch
import gym

gym.logger.set_level(40)
from api.evolution_strategies import GaussianMutationStrategy
from api.evaluators import ParallelEnvEvaluator
from api.deep_neuroevolution import GAOptimizer
from api.utils import Policy


# Define policy
class PongTorchPolicy(Policy, torch.nn.Module):
    # Input and output sizes from the HalfCheetah environment
    N_INPUTS = (210, 160, 3)  # Box(210, 160, 3)
    N_OUTPUTS = 6  # (discrete)

    def __init__(self, device=torch.device("cpu")):
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
        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = torch.nn.Linear(22528, 512)
        self.fc2 = torch.nn.Linear(512, 6)
        self.device = device
        self.to(device)

    def forward(self, x: torch.Tensor):
        # Expected input dimension: (?, 210, 160, 3)
        # Must move channels to 2. dim and create batch dimension if needed

        if x.ndimension() < 4:
            x.unsqueeze_(0)  # Add batch dimension

        x = x.permute(0, 3, 1, 2)  # Move channel to second dimension

        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flattens the input, except the batch dimension
        x = self.fc1(x)
        x = self.fc2(x)
        # x = torch.nn.functional.softmax(x)
        return x

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        # Observation to torch tensor, add empty batch dimension
        obs = torch.from_numpy(obs).float().to(self.device)
        with torch.no_grad():
            action = torch.argmax(self.forward(obs), dim=1)
        return action.detach().cpu().numpy()  # Back to numpy array and return


# Create environment factory
def env_factory() -> gym.Env:
    return gym.make('PongNoFrameskip-v4')


# Create policy factory
def policy_factory() -> Policy:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = PongTorchPolicy()
    # policy.share_memory()
    return policy


if __name__ == '__main__':

    gym.logger.set_level(40)
    # Create evaluator
    evaluator = ParallelEnvEvaluator(env_factory=env_factory, n_processes=8)
    # evaluator = EnvEvaluator(env_factory=env_factory)
    evolution_strategy = GaussianMutationStrategy(evaluator, policy_factory, generation_size=50, n_elites=10, n_check_top=5,
                                                  n_check_times=2)


    def eval_callback(results):
        rewards = [result[0] for result in results]
        print(np.mean(rewards), rewards[:10])


    # eval_callback = lambda results: print(np.mean(results), results[:10])
    optimizer = GAOptimizer(env_factory, policy_factory, evolution_strategy, evaluator, eval_callback=eval_callback)
    for _ in range(20):
        optimizer.train_generation()
