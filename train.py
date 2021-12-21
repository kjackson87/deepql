import argparse

import gym
import matplotlib.pyplot as plt

import agents
import utils

def run(args):
  env = gym.make('CartPole-v0')
  env.seed(args.seed)

  agent = agents.RandomAgent(env.action_space)

  rewards = []

  for i_episode in range(args.episodes):
    cur_observation = env.reset()
    reward = None
    episode_reward = 0
    for t in range(100):
      if args.render: env.render()
      action = agent.select_action(cur_observation)
      next_observation, reward, done, info = env.step(action)
      agent.update(cur_observation, action, reward, next_observation, done)
      cur_observation = next_observation
      episode_reward += reward
      if done:
        rewards.append(episode_reward)
        print(f'Episode {i_episode} finished after {t+1} timesteps with total reward {episode_reward}')
        break

  utils.plot_rewards(rewards)
  plt.show()
  agent.loss_meter.plot()
  plt.show()
  env.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='DeepQL Training Framework')
  parser.add_argument('-e', '--episodes', type=int,
                      default=20, help='Number of episodes to run')
  parser.add_argument('-r', '--render', action='store_true', help='Render or not')
  parser.add_argument('-s', '--seed', type=int, default=42, help='Seed')
  args = parser.parse_args()

  run(args)
