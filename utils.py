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

class Meter():
  def __init__(self, name):
    self.name = name
    self.reset()

  def reset(self):
    self.values = []

  def update(self, val):
    self.values.append(val)

  def plot(self, ax=None):
    if ax == None:
      fig, ax = plt.subplots()
    ax.plot(self.values)
    ax.set_title(self.name)
