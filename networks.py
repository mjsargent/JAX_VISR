import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import flax
import flax.linen as nn
import optax
import wandb
import time
import matplotlib.pyplot as plt
import pickle
import os
import utils

from flax.training import checkpoints
from flax.training.train_state import TrainState
from functools import partial
from typing import Sequence
from argparse import ArgumentParser

class TrainState(TrainState):
    target_params: flax.core.FrozenDict

# define models
class PsiModel(nn.Module):
  features: Sequence[int]

  @nn.compact
  def __call__(self, inputs):
    x = inputs
    for i, feat in enumerate(self.features):
      x = nn.Dense(feat)(x)
      if i != len(self.features) - 1:
        x = nn.relu(x)
    return x

class PhiModel(nn.Module):
  features: Sequence[int]

  @nn.compact
  def __call__(self, inputs):
    x = inputs
    for i, feat in enumerate(self.features):
      x = nn.Dense(feat)(x)
      if i != len(self.features) - 1:
        x = nn.relu(x)
      #else:
        #x = lambda x: x / jax.sqrt(max(sum(x**2), 1e-16))
        #x = x / jnp.sqrt(max(sum(x**2), 1e-16))
    return x

class CNNPsi(nn.Module):
  """A simple CNN model with concat w """
  phi_dim: int
  @nn.compact
  def __call__(self, x, w, lstm_state):
    x = nn.Conv(features=16, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = x.reshape((x.shape[0], -1))  # flatten
    x = jnp.concatenate((x,w), axis = -1)
    x = nn.Dense(features=128)(x)
    x = nn.relu(x)
    x = nn.Dense(features=self.phi_dim)(x)
    return x, lstm_state

class CNNPsiMultiBranch(nn.Module):
  """A simple CNN model with concat w, and multiple trunks for psi cummulants """
  phi_dim: int
  num_a: int
  @nn.compact
  def __call__(self, x, w, recurrent_state):
    x = nn.Conv(features=16, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = x.reshape((x.shape[0], -1))  # flatten
    x = jnp.concatenate((x,w), axis = -1)
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = jnp.concatenate([nn.Dense(features=self.phi_dim) \
                         (nn.relu(nn.Dense(features = 256)(x))) for i in range(self.num_a)] , axis = -1 )
    return x, recurrent_state

class MLPPsiMultiBranch(nn.Module):
  """A simple CNN model with concat w, multiple trunks for psi cummulants,
        and an LSTM core"""
  phi_dim: int
  num_a: int
  @nn.compact
  def __call__(self, x, w, lstm_state):
    # expects a tuple of (x, (h, c))
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = jnp.concatenate((x,w), axis = -1)
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=256)(x)
    x = jnp.concatenate([nn.Dense(features=self.phi_dim) \
                         (nn.relu(nn.Dense(features = 256)(x))) for i in range(self.num_a)] , axis = -1 )
    return x, lstm_state


class MLPLSTMPsiMultiBranch(nn.Module):
  """A simple CNN model with concat w, multiple trunks for psi cummulants,
        and an LSTM core"""
  phi_dim: int
  num_a: int
  lstm_size: int
  @nn.compact
  def __call__(self, x, w, lstm_state):
    # expects a tuple of (x, (h, c))
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = jnp.concatenate((x,w), axis = -1)
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    lstm_state, x = nn.OptimizedLSTMCell()(lstm_state, x)
    x = jnp.concatenate([nn.Dense(features=self.phi_dim) \
                         (nn.relu(nn.Dense(features = 256)(x))) for i in range(self.num_a)] , axis = -1 )
    return x, lstm_state


class CNNLSTMPsiMultiBranch(nn.Module):
  """A simple CNN model with concat w, multiple trunks for psi cummulants,
        and an LSTM core"""
  phi_dim: int
  num_a: int
  lstm_size: int
  @nn.compact
  def __call__(self, x, w, lstm_state):
    # expects a tuple of (x, (h, c))
    x = nn.Conv(features=16, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = x.reshape((x.shape[0], -1))  # flatten
    x = jnp.concatenate((x,w), axis = -1)
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    lstm_state, x = nn.OptimizedLSTMCell()(lstm_state, x)
    x = jnp.concatenate([nn.Dense(features=self.phi_dim) \
                         (nn.relu(nn.Dense(features = 256)(x))) for i in range(self.num_a)] , axis = -1 )
    return x, lstm_state

class MLPPsiMultiBranchTemporal(nn.Module):

  """A simple MLP model with concat w, multiple trunks for psi cummulants,
     multiple trunks for outputing expected psi in time"""
  phi_dim: int
  num_a: int
  lstm_size: int
  max_repeat: int
  @nn.compact
  def __call__(self, x, w, lstm_state):
    # expects a tuple of (x, (h, c))
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = jnp.concatenate((x,w), axis = -1)
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)

    psi_a = jnp.concatenate([nn.Dense(features=self.phi_dim) \
                         (nn.relu(nn.Dense(features = 256)(x))) for i in range(self.num_a)] , axis = -1 )
    q = utils.q_from_psi_vmap(psi_a, w)

    x = jnp.concatenate([x, q], axis = -1)

    psi_repeat= jnp.concatenate([nn.Dense(features=self.phi_dim) \
                         (nn.relu(nn.Dense(features = 256)(x))) for i in range(self.max_repeat)] , axis = -1 )
    return psi_a, psi_repeat, lstm_state


class MLPLSTMPsiMultiBranchTemporal(nn.Module):

  """A simple MLP model with concat w, multiple trunks for psi cummulants,
     multiple trunks for outputing expected psi in time, and an LSTM core"""
  phi_dim: int
  num_a: int
  lstm_size: int
  max_repeat: int
  @nn.compact
  def __call__(self, x, w, lstm_state):
    # expects a tuple of (x, (h, c))
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = jnp.concatenate((x,w), axis = -1)
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    lstm_state, x = nn.OptimizedLSTMCell()(lstm_state, x)

    psi_a = jnp.concatenate([nn.Dense(features=self.phi_dim) \
                         (nn.relu(nn.Dense(features = 256)(x))) for i in range(self.num_a)] , axis = -1 )
    q = utils.q_from_psi_vmap(psi_a, w)

    x = jnp.concatenate([x, q], axis = -1)

    psi_repeat= jnp.concatenate([nn.Dense(features=self.phi_dim) \
                         (nn.relu(nn.Dense(features = 256)(x))) for i in range(self.max_repeat)] , axis = -1 )
    return psi_a, psi_repeat, lstm_state


class CNNLSTMPsiMultiBranchTemporal(nn.Module):
  """A simple CNN model with concat w, multiple trunks for psi cummulants,
     multiple trunks for outputing expected psi in time, and an LSTM core"""
  phi_dim: int
  num_a: int
  lstm_size: int
  max_repeat: int
  @nn.compact
  def __call__(self, x, w, lstm_state):
    # expects a tuple of (x, (h, c))
    x = nn.Conv(features=16, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = x.reshape((x.shape[0], -1))  # flatten
    x = jnp.concatenate((x,w), axis = -1)
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    lstm_state, x = nn.OptimizedLSTMCell()(lstm_state, x)

    psi_a = jnp.concatenate([nn.Dense(features=self.phi_dim) \
                         (nn.relu(nn.Dense(features = 256)(x))) for i in range(self.num_a)] , axis = -1 )
    q = utils.q_from_psi_vmap(psi_a, w)

    x = jnp.concatenate([x, q], axis = -1)

    psi_repeat= jnp.concatenate([nn.Dense(features=self.phi_dim) \
                         (nn.relu(nn.Dense(features = 256)(x))) for i in range(self.max_repeat)] , axis = -1 )
    return psi_a, psi_repeat, lstm_state

class CNNPsiMultiBranchTemporal(nn.Module):
  """A simple CNN model with concat w, multiple trunks for psi cummulants,
     multiple trunks for outputing expected psi in time, and an core"""
  phi_dim: int
  num_a: int
  lstm_size: int
  max_repeat: int
  @nn.compact
  def __call__(self, x, w, lstm_state):
    # expects a tuple of (x, (h, c))
    x = nn.Conv(features=16, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = x.reshape((x.shape[0], -1))  # flatten
    x = jnp.concatenate((x,w), axis = -1)
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)

    psi_a = jnp.concatenate([nn.Dense(features=self.phi_dim) \
                         (nn.relu(nn.Dense(features = 256)(x))) for i in range(self.num_a)] , axis = -1 )
    q = utils.q_from_psi_vmap(psi_a, w)

    x = jnp.concatenate([x, q], axis = -1)

    psi_repeat= jnp.concatenate([nn.Dense(features=self.phi_dim) \
                         (nn.relu(nn.Dense(features = 256)(x))) for i in range(self.max_repeat)] , axis = -1 )
    return psi_a, psi_repeat, lstm_state


class MLPPhi(nn.Module):
  """A simple MLP model """
  output_size: int
  @nn.compact
  def __call__(self, x):
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=self.output_size)(x)
    return x


class CNNPhi(nn.Module):
  """A simple CNN model """
  output_size: int
  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=16, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=self.output_size)(x)
    return x



