import torch
import torchvision
from torchvision import datasets, transforms
import torch.utils.data
import numpy as np
import gym
from multiprocessing import Pool

gym.logger.set_level(40)
device = torch.device("cpu")
import random


def eval_model(model, obs_means, obs_stds, times=1, env=None, ):
    if env is None:
        env = gym.make("HalfCheetah-v2")
    rewards = []
    for _ in range(times):
        obs = env.reset()
        done = False
        total_reward = 0
        i = 0
        while not done:
            obs = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                action = torch.distributions.Normal(model.forward(obs), 0.1).sample()
            obs, reward, done, _ = env.step(action)
            obs = (obs - obs_means) / (obs_stds + 1e-7)
            total_reward += reward
        rewards.append(total_reward)
    return (np.mean(rewards), model)


def create_model():
    return torch.nn.Sequential(
        torch.nn.Linear(N_INPUTS, 16),
        torch.nn.Tanh(),
        torch.nn.Linear(16, 16),
        torch.nn.Tanh(),
        torch.nn.Linear(16, N_OUTPUTS)
    ).to(device)


def gen_population(elites, n_models):
    offspring = []
    for i in range(n_models):
        parent = elites[random.randint(0, len(elites) - 1)]
        parent_state_dict = parent.state_dict()
        model = create_model()
        model.load_state_dict(parent_state_dict)
        for tensor in model.state_dict().values():
            mutation = torch.randn_like(tensor) * 0.02
            tensor += mutation
        offspring.append(model)
    return offspring


# obs_means = None
# obs_stds = None
if __name__ == '__main__':
    global obs_means
    global obs_stds
    with Pool(16) as p:
        # print(p.map(f, [1, 2, 3]))
        env = gym.make("HalfCheetah-v2")
        N_INPUTS = 17
        N_OUTPUTS = 6
        done = False
        obs = env.reset()
        obs_dataset = [obs]

        # Normalization statistics
        for _ in range(1000):
            action = random.randint(0, 1)
            obs_dataset.append(obs)
            obs, reward, done, _ = env.step(action)
            if done:
                obs = env.reset()

        obs_dataset = np.array(obs_dataset)
        obs_means = np.mean(obs_dataset, axis=0)
        obs_stds = np.std(obs_dataset, axis=0)

        elite = create_model()
        # Make bias 0
        for key in elite.state_dict():
            if key.endswith("bias"):
                elite.state_dict()[key] *= 0

        # Initial population
        offspring = gen_population([elite], 500)
        jobs = [(o, obs_means, obs_stds) for o in offspring]
        results = p.starmap(eval_model, jobs)  # [eval_model(o, env) for o in offspring]

        # Run for generations
        for _ in range(200):
            elites = sorted(results, key=lambda x: x[0], reverse=True)[:20]
            rewards = [e[0] for e in elites]
            print(np.mean([r[0] for r in results]), rewards)  # print rewards

            elites = [elite[1] for elite in elites]
            elites_checked = p.starmap(eval_model, [(e, obs_means, obs_stds, 20) for e in elites[:10]])
            elites_checked = sorted(elites_checked, key=lambda x: x[0], reverse=True)

            offspring = gen_population(elites, 999)
            offspring.append(elites_checked[0][1])

            results = p.starmap(eval_model, [(o, obs_means, obs_stds) for o in offspring])
            # results = [eval_model(o, env) for o in offspring]
        # print([o.state_dict()["0.bias"] for o in offspring])
        # eval_model(model)
