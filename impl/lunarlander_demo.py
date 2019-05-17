import argparse

import gym
import torch
#from impl.lunar_lander_minimal import LunarLanderTorchPolicy
import pygame
from impl.play import play

class LunarLanderTorchPolicy(torch.nn.Module):
    N_INPUTS = 8  # env.action_space.shape
    N_OUTPUTS = 4  # env.observation_space.shape

    def __init__(self):
        super().__init__()
        # self.obs_normalizer = obs_normalizer
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

    def __call__(self, obs):
        # Observation to torch tensor, add empty batch dimension
        obs = torch.from_numpy(obs).float().unsqueeze(0)
        action = self.net.forward(obs)
        return torch.argmax(action, dim=1).detach().numpy()[0]  # Back to numpy array and return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="model_state_dict.pth")
    args = parser.parse_args()

    model_path = args.path
    policy = LunarLanderTorchPolicy()
    policy.load_state_dict(torch.load(model_path))
    env = gym.make("LunarLander-v2")
    done = False
    obs = env.reset()
    play(env, zoom=4, fps=60, get_action_fn=policy, keys_to_action={(119,): 0, (97,): 1, (115,): 2, (100,): 3})
    """
    while True:
        env.render()

        for event in pygame.event.get():
            # test events, set key states
            if event.type == pygame.KEYDOWN:
                print(event.key)

        obs, reward, done, info = env.step(policy(obs))
        if done:
            obs = env.reset()
    """



