import argparse
import gym
import io
import matplotlib.pyplot as plt
import numpy as np
import torch

from PIL import Image
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
        base_depth=32,
        layer_size=128,
        memory_capacity=10000,
        videos=True,
        video_freq=2):
    env = gym.make(environment_name)
    obs = env.reset()
    observation_shape = preprocess(obs).shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device: {}".format(device))
    agent = CategoricalAgent(
            (stack_size, *observation_shape),
            env.action_space,
            N=N,
            lr=step_size,
            v_min=v_min,
            v_max=v_max,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            feature_size=feature_size,
            base_depth=base_depth,
            layer_size=layer_size)
    replay = UniformExperienceReplay(memory_capacity,
                                     observation_shape,
                                     stack_size=stack_size,
                                     device=device)

    writer = SummaryWriter(logdir)

    print("Beginning initial collection (random actions)")
    obs = preprocess(obs)
    for _ in range(initial_collect_length):
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        next_obs = preprocess(next_obs)
        replay.add(obs, action, reward, next_obs, 1 if done else 0)
        if done:
            obs = preprocess(env.reset())
        else:
            obs = next_obs

    obs = preprocess(env.reset())
    episode_length = 0
    episode_reward = 0
    episode_number = 0
    print("Training begins...")
    obs_state = np.zeros((stack_size, *observation_shape))
    writer.add_graph(agent, torch.tensor(obs_state).to(device))
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
            if max_episode_steps is not None:
                episode_too_long = episode_length >= max_episode_steps
            else:
                episode_too_long = False
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
        loss, artifacts = agent.train_agent(experience)
        artifacts.update({'loss': loss})
        if videos and (i % video_freq == 0):
            video = gen_video(
                    agent,
                    environment_name,
                    stack_size,
                    observation_shape,
                    max_episode_steps)
            artifacts.update({'video': video})
        update_epoch_summaries(writer, agent, artifacts, i)

def gen_video(agent, env_name, stack_size, observation_shape, max_frames):
    frames = []
    env = gym.make(env_name)
    obs = env.reset()
    frames.append(obs)
    obs = preprocess(obs)
    obs_state = np.zeros((stack_size, *observation_shape))
    episode_length = 0
    while True:
        episode_length += 1
        obs_state = np.concatenate(([obs], obs_state[:-1]), axis=0)
        action = agent.action(obs_state, explore=False)
        obs, reward, done, _ = env.step(action)
        frames.append(obs)
        obs = preprocess(obs)
        if done:
            break
        if max_frames is None:
            continue
        if episode_length > max_frames:
            break
    return frames

def update_epoch_summaries(writer, agent, artifacts, epoch):
    writer.add_scalar('Training/Loss', artifacts['loss'], epoch)
    writer.add_scalar('Training/Epsilon', artifacts['epsilon'], epoch)
    writer.add_scalar('Training/Reward_Density', artifacts['reward_density'], epoch)
    writer.add_images('Obs/Images', artifacts['images'].unsqueeze(1), dataformats="NCHW")
    value_dist_img = gen_value_dist_plot(agent.value_support(), artifacts['value_distribution'])
    writer.add_image('Obs/Distributions', torch.tensor(value_dist_img), dataformats='HWC')
    if 'video' in artifacts.keys():
        video = torch.tensor(artifacts['video']).permute(0, 3, 1, 2).unsqueeze(0)
        writer.add_video('Preview', video, fps=30)

def gen_value_dist_plot(value_support, value_dist):
    buf = io.BytesIO()
    support = value_support.detach().cpu().numpy()
    for i in range(value_dist.shape[0]):
        dist = value_dist[i].detach().cpu().numpy()
        plt.bar(support, dist, alpha=0.5, label="Action {}".format(i))
    plt.legend()
    plt.savefig(buf)
    plt.clf()
    img = np.asarray(Image.open(buf).convert('RGB'))
    return img

def preprocess(obs):
    # Convert to grayscale
    obs = np.mean(obs, axis=2).astype(np.uint8)
    # Downsample to save memory
#    obs = obs[::2, ::2]
    return obs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='logs', help='Directory for tensorboard logs')
    parser.add_argument('--env', type=str, default='Breakout-v0', help='Gym environment name')
    parser.add_argument('--epochs', type=int, default=1000, help='Number training epochs')
    parser.add_argument('--steps_per_epoch', type=int, default=1000, help='Environment steps per epoch')
    parser.add_argument('--epsilon', type=float, default=0.05, help='For e-greedy')
    parser.add_argument('--epsilon_decay', type=float, default=1e-4, help='For e-greedy')
    parser.add_argument('--feature_size', type=int, default=128)
    parser.add_argument('--base_depth', type=int, default=128)
    parser.add_argument('--initial_collect_length', type=int, default=1000)
    parser.add_argument('--memory_capacity', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--step_size', type=float, default=3e-4)
    parser.add_argument('--layer_size', type=int, default=128)
    parser.add_argument('--atoms', type=int, default=51)
    parser.add_argument('--no_video', default=False, action="store_true", help="Add video previews to tensorboard")
    parser.add_argument('--video_freq', type=int, default=10)
    args = parser.parse_args()
    train_categorical_agent(
            args.env,
            logdir=args.logdir,
            num_epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
            feature_size=args.feature_size,
            base_depth=args.base_depth,
            memory_capacity=args.memory_capacity,
            initial_collect_length=args.initial_collect_length,
            batch_size=args.batch_size,
            layer_size=args.layer_size,
            step_size=args.step_size,
            N=args.atoms,
            videos=not args.no_video,
            video_freq=args.video_freq)
