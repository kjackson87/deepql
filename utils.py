import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(rewards):
  plt.xlabel('Episode')
  plt.ylabel('Reward')
  plt.plot(rewards)
  plt.plot(moving_average(rewards, 5))

def moving_average(data, window):
  start_window = np.full(window - 1, np.nan)
  mva = np.append(start_window, np.convolve(data, np.ones(window), 'valid') / window)
  return mva

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
    return ax

class SMAMeter(Meter):
  def __init__(self, name, window=5):
      super().__init__(name)
      self.window = window
      self.averages = []

  def update(self, val):
    super().update(val)
    if len(self.values) < self.window:
      self.averages.append(None)
    else:
      self.averages.append(np.average(self.values[-self.window:]))

  def plot(self, ax=None):
    ax = super().plot(ax)
    ax.plot(self.averages)