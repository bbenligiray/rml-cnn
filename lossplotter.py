# from https://gist.github.com/stared/dfb4dfaf6d9a8501cd1cc8b8cb806d2e

import os

import keras
from matplotlib import pyplot as plt


class LossPlotter(keras.callbacks.Callback):
  def __init__(self, log_path='log'):
    self.log_path = log_path
    
  def on_train_begin(self, logs={}):
    self.i = 0
    self.x = []
    self.losses = []
    self.val_losses = []
        
    self.fig = plt.figure()
        
    self.logs = []
    plt.ion();

  def on_epoch_end(self, epoch, logs={}):
    self.logs.append(logs)
    self.x.append(self.i)
    self.losses.append(logs.get('loss'))
    self.val_losses.append(logs.get('val_loss'))
    self.i += 1
        
    plt.close()        
    plt.plot(self.x, self.losses, label="loss")
    plt.plot(self.x, self.val_losses, label="val_loss")
    plt.legend()
    plt.pause(0.01)
    plt.savefig(self.log_path)
