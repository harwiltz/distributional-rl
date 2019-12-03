import argparse
import gym
import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter

from distrl.categorical.categorical_agent import CategoricalAgent
from distrl.utils.replay import UniformExperienceReplay

def train_categorical_agent(
        environment_name,
        logdir='logs',
        initial_collect_length=1000,
        num_epochs=1000,
        batch_size=32,
        steps_per_epoch=1000,
        max_episode_steps=None,
        gamma=0.99,
        step_size=3e-4,
        N=51,
        v_min=-10,
        v_max=10,
        epsilon=0.05,
        epsilon_decay=0.001,
        feature_size=128,
        base_depth=128,
        layer_size=128,
        memory_capacity=10000):
    env = gym.make(environment_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device: {}".format(device))
    agent = CategoricalAgent(
            env.observation_space,
            env.action_space,
            N=N,
            v_min=v_min,
            v_max=v_max,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            feature_size=feature_size,
            base_depth=base_depth,
            layer_size=layer_size)
    replay = UniformExperienceReplay(memory_capacity, env.observation_space.shape, device)

    writer = SummaryWriter(logdir)

    obs = env.reset()
    print("Beginning initial collection (random actions)")
    for _ in range(initial_collect_length):
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        replay.add(obs, action, reward, next_obs, 1 if done else 0)

    obs = env.reset()
    episode_length = 0
    episode_reward = 0
    episode_number = 0
    print("Training begins...")
    for i in range(num_epochs):
        done = False
        for j in range(steps_per_epoch):
            episode_length += 1
            formatted_obs = format_obs(obs, device)
            action = agent.action(formatted_obs)
            next_obs, reward, done, _ = env.step(action)
            episode_reward += reward
            replay.add(obs, action, reward, next_obs, 1 if done else 0)
            episode_too_long = (max_episode_steps is not None) and (j >= max_episode_steps)
            if done or episode_too_long:
                obs = env.reset()
                writer.add_scalar('Performance/Score', episode_reward, episode_number)
                writer.add_scalar('Performance/Episode_Length', episode_length, episode_number)
                episode_length = 0
                episode_reward = 0
                episode_number += 1
            else:
                obs = next_obs
        experience = replay.sample(batch_size)
        loss = agent.train(experience)
        writer.add_scalar('Training/Loss', loss, i)

def format_obs(obs, device):
    obs = torch.FloatTensor(obs).to(device) / float(255)
    return obs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='logs', help='Directory for tensorboard logs')
    parser.add_argument('--env', type=str, default='Breakout-v0', help='Gym environment name')
    parser.add_argument('--epochs', type=int, default=1000, help='Number training epochs')
    parser.add_argument('--steps_per_epoch', type=int, default=1000, help='Environment steps per epoch')
    args = parser.parse_args()
    train_categorical_agent(
            args.env,
            logdir=args.logdir,
            num_epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch)
