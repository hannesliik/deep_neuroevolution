import argparse
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
from api.utils import Policy

# Disable annoying warnings from gym
gym.logger.set_level(40)


# Define policy
class LunarLanderTorchPolicy(Policy, torch.nn.Module):
    N_INPUTS = 8  # env.action_space.shape
    N_OUTPUTS = 4  # env.observation_space.shape

    def __init__(self):
        super().__init__()
        #self.obs_normalizer = obs_normalizer
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
        #obs = self.obs_normalizer.normalize(obs)
        obs = torch.from_numpy(obs).float().unsqueeze(0)
        action = self.net.forward(obs)
        return torch.argmax(action, dim=1).detach().numpy()[0]  # Back to numpy array and return


# Create environment factory
def env_factory() -> gym.Env:
    return gym.make("LunarLander-v2")


#obs_normalizer = ObsNormalizer(env_factory, n_samples=2000)


# Create policy factory
def policy_factory() -> Policy:
    policy = LunarLanderTorchPolicy()
    # Tell torch that we will not calculate gradients.
    # Up to ~10% speedup and maybe takes less memory
    for param in policy.parameters():
        param.requires_grad = False
    return policy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str)
    parser.add_argument("-g", "--gen_size", type=int, default=200)
    parser.add_argument("-e", "--elites", type=int, default=20)
    parser.add_argument("-cn", "--check_n", type=int, default=10)
    parser.add_argument("-ct", "--check_times", type=int, default=30)
    parser.add_argument("-d", "--decay", type=float, default=1)
    parser.add_argument("-std", type=float, default=0.1)
    parser.add_argument("-i", "--iterations", type=int, default=50)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--path", type=str, default="data")
    parser.add_argument("-ps", "--parent_selection", type=str, choices=['uniform', 'probab'], default='uniform')
    parser.add_argument("-t", "--times", type=int, default=1,
                        help="The average of t runs is the evaluation score of a policy")

    args = parser.parse_args()
    args = vars(args)
    assert args["check_n"] <= args["elites"]
    assert args["gen_size"] > args["elites"]
    assert args["times"] >= 1
    assert args["decay"] > 0

    # Create evaluator
    evaluator = ParallelEnvEvaluator(env_factory=env_factory, times=args["times"])

    env = env_factory()

    '''
    evolution_strategy = GaussianMutationStrategy(policy_factory, evaluator=evaluator,
                                                  parent_selection=args["parent_selection"],
                                                  std=args["std"],
                                                  size=args["gen_size"], n_elites=args["elites"],
                                                  n_check_top=args["check_n"], n_check_times=args["check_times"],
                                                  decay=args["decay"])
    '''
    evolution_strategy = CrossoverStrategy(policy_factory, evaluator=evaluator,
                                            parent_selection=args["parent_selection"],
                                            size=args["gen_size"], n_elites=args["elites"])

    optimizer = GAOptimizer(policy_factory, evolution_strategy)

    experiment_name = args["exp_name"] + "_" + time.strftime("%Y%m%d_%H%M%S") + "_lunar_lander"

    if not args["path"].endswith("/"):
        args["path"] += "/"
    if not os.path.exists(args["path"]):
        os.makedirs(args["path"])
    if not os.path.exists(args["path"] + experiment_name):
        os.makedirs(args["path"] + experiment_name)
    prefix = args["path"] + experiment_name + "/"

    with open(prefix + "params.json", "w") as fp:
        #json.dump(evolution_strategy.state["params"], fp)
        json.dump(args, fp)

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
        df.to_csv(prefix + "plot.csv")
        if args["plot"]:
            sns.lineplot(data=df, x="time", y="score", ci="sd")
            plt.savefig(prefix + f"plot_{i}.png")
