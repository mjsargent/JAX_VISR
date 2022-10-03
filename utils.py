import jax
import jax.numpy as jnp
import numpy as np
import gymnax

import flax
import flax.linen as nn
import optax
import networks

from flax.training import checkpoints
from flax.training.train_state import TrainState
from functools import partial
from typing import Sequence
from argparse import ArgumentParser


def progress(value, max=100):
    return HTML("""
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(value=value, max=max))

def sample_sphere(num_dim, size=None):
  if size is None:
    size = [traj_len, mb_size]
  unnormed = np.random.randn(*(size+[num_dim]))
  return unnormed / np.linalg.norm(unnormed, axis=-1, keepdims=true, ord = 2)

def reset(size, num_states, start_state=None):
  if start_state is None:
    state = np.random.randint(num_states, size=size)
  else:
    state = start_state*np.ones([size], dtype=np.int)
  return state

def get_obs(state):
  return np.eye(num_states)[state]

def get_rc(state, grid_size):
  row = state//grid_size
  col = state % grid_size
  return row, col

def get_s(row, col, grid_size):
  return row*grid_size+col

def step(state, action, grid_size):
  row, col = get_rc(state, grid_size)
  #a0:up
  new_r = np.where(action == 0, row-1, row)
  #a1:left
  new_c = np.where(action == 1, col-1, col)
  #a2:down
  new_r = np.where(action == 2, new_r+1, new_r)
  #a3:right
  new_c = np.where(action == 3, new_c+1, new_c)
  #clip
  new_r[new_r >= grid_size] = grid_size-1
  new_r[new_r < 0] = 0
  new_c[new_c >= grid_size] = grid_size-1
  new_c[new_c < 0] = 0
  return get_s(new_r, new_c, grid_size)

# VISR functions
@jax.jit
def norm_sphere(unnormed_sphere):
    norm = jnp.linalg.norm(unnormed_sphere, axis = -1, keepdims = True, ord = 2)
    return unnormed_sphere / norm

def sample_uniform_sphere(shape, rng_key):
  unnormed = jax.random.normal(shape = shape, key = rng_key)
  return norm_sphere(unnormed)

# q function for acting
@jax.jit
def q_from_psi(psi, w):
  # psi is A*pi)
  # TODO need to reshape
  psi = psi.reshape( -1, w.shape[-1])
  q = jnp.sum(jnp.multiply(psi, w), -1)
  return q

q_from_psi_vmap = jax.vmap(q_from_psi)
q_from_single_psi = jax.jit(jax.vmap(q_from_psi, in_axes = [None, 0]))

# select psi given an action was taken
@partial(jax.jit, static_argnames = ["phi_dim"])
def select_psi_from_action(psi, action, phi_dim):
  # psi is (A x |phi|) -> want |phi|
  selected_psi = psi.reshape(-1, phi_dim)
  return selected_psi[action, :]

#
psi_selector = jax.vmap(select_psi_from_action, in_axes = [0, 0, None])

@partial(jax.jit, static_argnames = ["gamma"])
def psi_td_error(psi, t_psi, phi, gamma):
  target = gamma * t_psi + phi
  #return (jax.lax.stop_gradient(target) - psi) **2
  return (target - psi) **2

@partial(jax.jit, static_argnames = ["num_states"])
def obs_from_s(s: jnp.ndarray, num_states: int):
  """
  convert 1-D array of indexes of states to a batch of one-hot embeddings
  """
  obs = jnp.zeros(num_states)
  obs = obs.at[s].set(1)
  # want: B X n_states
  return obs

obs_from_s_batched = jax.vmap(obs_from_s, in_axes = [0, None])

@jax.jit
def cat_w(w, w_gpi):
    w_gpi = jnp.concatenate((w[jnp.newaxis, :], w_gpi), axis = 0)
    return w_gpi

#@jax.jit
def normalise_phi(phi):
  """
  function for l2 normalising phi
  """
  #phi = phi / jnp.linalg.norm(phi, axis = -1, keepdims = True, ord = 2)
  phi = phi / (jax.lax.max(jnp.sum(phi**2, -1, keepdims = True), 1e-12 )**(0.5))
  #sqrt(max(sum(x**2), epsilon))
  return phi

# static argnum for  idx - not too many possible values for this so we'll eat
# the recompile time
@partial(jax.jit, static_argnums = [1,])
def idx_trajectory(trajectory: list, idx: int):
    """
    take a list of containers  and return a new list
    consisting of each element indexed at a given idx
    """
    def _idx(t, i):
        if isinstance(t, tuple):
            f_t = tuple([_idx(_t, i )  for _t in t])
            return f_t
        else:
            return t[i]
    return [_idx(t, idx) for t in trajectory]

@partial(jax.jit, static_argnums = [1,])
def slice_trajectory(trajectory: list, idx: int):
    """
    take a list of containers  and return a new list
    consisting of each a slice of each element from idx onwards
    """
    def _slice(t, i):
        if isinstance(t, tuple):
            f_t = tuple([_slice(_t, i )  for _t in t])
            return f_t
        else:
            return t[i:]
    return [_slice(t, idx) for t in trajectory]

def flatten_trajectory(trajectory:list , num_batch_dims: int =  2):
    """
    take a list of containers  and return a new list
    consisting of each element flattened
    """
    def _flatten(t, n_dims):
        if isinstance(t, tuple):
            f_t = tuple([_flatten(_t, n_dims) for _t in t])

            return f_t
        else:
            return t.reshape(-1, *t.shape[n_dims:])
    return [_flatten(_t, num_batch_dims) for _t in trajectory]

def bool_idx_trajectory(trajectory: list, valid: jnp.ndarray):
    """
    take a list of JAX arrays  and return a new list
    consisting of arrays corrosponding with the elements
    of the previous arrays specified by the bool mask
    """
    valid = jnp.array(valid, dtype = int)
    def _bool_idx(t, v):
        if isinstance(t, tuple):
            f_t = tuple([_bool_idx(_t, v) for _t in t])
            return f_t
        else:
            return  t[valid == 1]

    return [_bool_idx(t, valid) for t in trajectory]


def concat_trajectory(acc_trajectory: list, partial_trajectory: list):
    """
    take two lists of JAX arrays and return a new list
    consisting of the concatenation of the elements of each
    """
    def _concat(a, p):
        if isinstance(a, tuple):
            f_t = tuple([_concat(_a, _p )  for _a, _p in zip(a, p)])
            return f_t
        else:
            return jnp.concatenate((a, p), axis = 0)
    return [_concat(a,p) for a, p in zip (acc_trajectory, partial_trajectory)]

def make_networks(game: str, recurrent: str, phi_dim:int, num_actions: int, lstm_size: int = 128):
    if recurrent:
        if game in ["Breakout", "SpaceInvaders", "Freeway", "Asterix", "Seaquest", "Catch"]:
            psi_net = networks.CNNLSTMPsiMultiBranch(
            phi_dim=phi_dim, num_a=num_actions, lstm_size=lstm_size
            )
            phi_net = networks.CNNPhi(output_size=phi_dim)

        else:
            psi_net = networks.MLPLSTMPsiMultiBranch(
            phi_dim=phi_dim, num_a=num_actions, lstm_size=lstm_size
            )
            phi_net = networks.MLPPhi(output_size=phi_dim)
    else:

        if game in ["Breakout", "SpaceInvaders", "Freeway", "Asterix", "Seaquest", "Catch"]:
            psi_net = networks.CNNPsiMultiBranch(phi_dim=phi_dim, num_a=num_actions)
            phi_net = networks.CNNPhi(output_size=phi_dim)
        else:

            psi_net = networks.MLPPsiMultiBranch(phi_dim=phi_dim, num_a=num_actions)
            phi_net = networks.MLPPhi(output_size=phi_dim)


    return phi_net, psi_net

def make_env(game: str):

    if game in ["Breakout", "SpaceInvaders", "Freeway", "Asterix", "Seaquest"]:
        game_str = game + "-MinAtar"
        env, env_params = gymnax.make(game_str)
        num_a = len(env.action_set)

    elif game in ["DeepSea", "Catch"]:
        game_str = game + "-bsuite"
        env, env_params = gymnax.make(game_str)
        num_a = env.action_space().n

    elif game in ["CartPole"]:
        game_str = game + "-v1"
        env, env_params = gymnax.make(game_str)
        num_a = env.action_space().n

    #elif args.game in ["Acrobot", "Pendulum"]:
    #    game_str = args.game + "-v1"
    #    env, env_params = gymnax.make(game_str)
    #   num_a = env.action_space().


    elif game in ["MountainCar"]:
        game_str = game + "-v0"
        env, env_params = gymnax.make(game_str)
        num_a = env.action_space().n
    else:
        game_str = game + "-misc"
        env, env_params = gymnax.make(game_str)
        num_a = env.action_space().n


    return env, env_params, num_a, game_str
