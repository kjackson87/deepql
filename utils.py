import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(rewards):
  plt.xlabel('Episode')
  plt.ylabel('Reward')
  plt.plot(rewards)
  plt.plot(moving_average(rewards, 5))

def moving_average(data, window):
  start_window = np.full(window, np.nan)
  return np.append(start_window, np.convolve(data, np.ones(window), 'valid') / window)
