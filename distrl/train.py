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
        stack_size=4,
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
    obs = env.reset()
    observation_shape = preprocess(obs).shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device: {}".format(device))
    agent = CategoricalAgent(
            (stack_size, *observation_shape),
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
    replay = UniformExperienceReplay(memory_capacity, observation_shape, device)

    writer = SummaryWriter(logdir)

    print("Beginning initial collection (random actions)")
    for _ in range(initial_collect_length):
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        replay.add(preprocess(obs), action, reward, preprocess(next_obs), 1 if done else 0)

    obs = preprocess(env.reset())
    episode_length = 0
    episode_reward = 0
    episode_number = 0
    print("Training begins...")
    obs_state = np.zeros((stack_size, *observation_shape))
    for i in range(num_epochs):
        done = False
        for j in range(steps_per_epoch):
            episode_length += 1
            obs_state = np.concatenate(([obs], obs_state[:-1]), axis=0)
            action = agent.action(obs_state)
            next_obs, reward, done, _ = env.step(action)
            next_obs = preprocess(next_obs)
            episode_reward += reward
            replay.add(obs, action, reward, next_obs, 1 if done else 0)
            episode_too_long = (max_episode_steps is not None) and (j >= max_episode_steps)
            if done or episode_too_long:
                obs = preprocess(env.reset())
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

def preprocess(obs):
    # Convert to grayscale
    obs = np.mean(obs, axis=2).astype(np.uint8)
    # Downsample to save memory
    obs = obs[::2, ::2]
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
