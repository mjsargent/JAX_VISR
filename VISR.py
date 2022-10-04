# imports
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import flax
import flax.linen as nn
import optax
import time
import matplotlib.pyplot as plt
import pickle
import os
import gymnax

# from gymnax.experimental import RolloutWrapper
from flax.training import checkpoints
from flax.training.train_state import TrainState
from functools import partial
from typing import Sequence
from argparse import ArgumentParser
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union
from distutils.util import strtobool

import utils
from networks import TrainState

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any

# argparse
parser = ArgumentParser("VISR over MinAtar envs")
parser.add_argument("--num_gpi_samples", default=10, type=int)
parser.add_argument("--traj_len", default=40, type=int)
parser.add_argument("--mb_size", default=16, type=int)
parser.add_argument("--hid_dim", default=100, type=int)
parser.add_argument("--phi_dim", default=5, type=int)
parser.add_argument("--gamma", default=0.95, type=float)
parser.add_argument("--epsilon", default=0.1, type=float)
parser.add_argument("--lr", default=3e-4, type=float)
parser.add_argument("--num_steps", default=30000, type=int)
parser.add_argument("--print_frequency", default=1000, type=int)
parser.add_argument("--target_update_frequency", default=100, type=int)
parser.add_argument("--eval_frequency", default=2500, type=int)
parser.add_argument("--log_frequency", default=500, type=int)
parser.add_argument("--overwrite", default=False, type=bool)
parser.add_argument("--gpi", default=True, type=bool)
parser.add_argument(
    "--game",
    default="Breakout",
    choices=[
        "Breakout",
        "Freeway",
        "SpaceInvaders",
        "Asterix",
        "CartPole",
        "MountainCar",
        "FourRooms",
        "DeepSea",
        "Catch",
    ],
    type=str,
)
parser.add_argument("--num_envs", default=16, type=int)
parser.add_argument("--num_eval_envs", default=16, type=int)
parser.add_argument("--num_rewarded_steps", default=100000, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--recurrent", default=False, type=lambda x: bool(strtobool(x)))
parser.add_argument("--lstm_size", default=128, type=int)
parser.add_argument("--log_wandb", default=False, type=lambda x: bool(strtobool(x)))
parser.add_argument(
    "--wandb_entity", type=str, help="user or group to log to if using wandb"
)
parser.add_argument(
    "--wandb_project", type=str, help="project to log to if using wandb"
)


args = parser.parse_args()

env, env_params, num_actions, game_str = utils.make_env(args.game)
args.game_str = game_str
# init environment


def training_loop(args, env, env_params):

    num_gpi_samples = args.num_gpi_samples
    traj_len = args.traj_len
    phi_dim = args.phi_dim
    gamma = args.gamma
    epsilon = args.epsilon
    lr = args.lr
    num_steps = args.num_steps
    print_frequency = args.print_frequency
    target_update_frequency = args.target_update_frequency
    eval_frequency = args.eval_frequency
    log_frequency = args.log_frequency
    num_envs = args.num_envs
    num_eval_envs = args.num_eval_envs
    num_rewarded_steps = args.num_rewarded_steps
    recurrent = args.recurrent
    lstm_size = args.lstm_size
    game = args.game
    log_wandb = args.log_wandb

    if log_wandb:
        import wandb

        entity = args.wandb_entity
        project = args.wandb_project
        wandb.init(entity=entity, project=project, config=args)

    w = jnp.zeros([1, num_envs, phi_dim])
    rng_key = jax.random.PRNGKey(args.seed)
    obs, state = env.reset(rng_key)
    obs_shape = obs.shape
    obs_shape = obs.shape
    init_obs = obs[jnp.newaxis]
    init_w = w[0, 0, :].copy()
    init_w = init_w[jnp.newaxis]
    init_psi_obs = obs[jnp.newaxis]

    phi_net, psi_net = utils.make_networks(
        game, recurrent, phi_dim, num_actions, lstm_size
    )
    print(psi_net)
    print(phi_net)

    psi_key, _ = jax.random.split(rng_key)
    phi_key, _ = jax.random.split(psi_key)

    init_recurrent_state = nn.OptimizedLSTMCell().initialize_carry(
        rng_key, batch_dims=(num_envs,), size=lstm_size
    )
    # init psi net with n_s + phi_dim for concat of w
    phi_state = TrainState.create(
        apply_fn=phi_net.apply,
        params=phi_net.init(phi_key, init_obs),
        target_params=phi_net.init(phi_key, init_obs),
        tx=optax.adam(learning_rate=10 * lr, eps=1e-5),
    )

    psi_state = TrainState.create(
        apply_fn=psi_net.apply,
        params=psi_net.init(psi_key, init_psi_obs, init_w, init_recurrent_state),
        target_params=psi_net.init(psi_key, init_psi_obs, init_w, init_recurrent_state),
        tx=optax.adam(learning_rate=lr, eps=1e-5),
    )
    psi_net.apply = jax.jit(psi_net.apply)
    phi_net.apply = jax.jit(phi_net.apply)

    def repeat_nested_recurrent_state(
        recurrent_state: Tuple[Array, Array], n_repeat: int
    ):
        r1, r2 = recurrent_state
        repeat_fn = lambda r, n: jnp.repeat(r, n, axis=0)
        r1 = repeat_fn(r1, n_repeat)
        r2 = repeat_fn(r2, n_repeat)
        return (r1, r2)

    def flatten_nested_recurrent_state(recurrent_state: Tuple[Array, Array], offset=1):
        r1, r2 = recurrent_state
        flatten_fn = lambda r: r.reshape(-1, *r.shape[2:])
        r1 = flatten_fn(r1)
        r2 = flatten_fn(r2)
        return (r1, r2)

    @jax.jit
    def gpi_act(psi_state, obs, w, recurrent_state, rng_key):
        obs = obs[jnp.newaxis]
        w = w[jnp.newaxis, :]

        psi, recurrent_state = psi_net.apply(psi_state.params, obs, w, recurrent_state)
        # psi: Bx (Ax|phi|)

        @partial(jax.jit, static_argnames=["n_samples"])
        def gpi_sample(_w, obs, n_samples, rng_key, recurrent_state):
            # psi: (A x |phi|)
            w_gpi = utils.sample_uniform_sphere((n_samples, _w.shape[-1]), rng_key)
            w_gpi = utils.cat_w(_w, w_gpi)
            if recurrent != None:
                repeated_recurrent_state = repeat_nested_recurrent_state(
                    recurrent_state, n_samples + 1
                )

            psi, _ = psi_net.apply(
                psi_state.params,
                jnp.repeat(obs[jnp.newaxis], n_samples + 1, axis=0),
                w_gpi,
                repeated_recurrent_state,
            )
            return psi.reshape(psi.shape[0], -1, phi_dim)

        psi = jax.vmap(gpi_sample, in_axes=[0, 0, None, None, None])(
            w, obs, num_gpi_samples, rng_key, recurrent_state
        )
        return psi, (recurrent_state[0].squeeze(), recurrent_state[1].squeeze())

    @jax.jit
    def get_gpi_action(psi, w):
        psi = psi.squeeze(0)
        w = w[jnp.newaxis]
        cur_q = utils.q_from_psi_vmap(
            psi, jnp.repeat(w, repeats=num_gpi_samples + 1, axis=0)
        )
        best_actions = jnp.argmax(cur_q, -1)
        best_values = jnp.max(cur_q, -1)
        best_gpi = jnp.argmax(best_values, -1)
        greedy_a = best_actions[best_gpi]
        return greedy_a

    @jax.jit
    def update(
        psi_state,
        phi_state,
        obs,
        actions,
        next_obs,
        w,
        recurrent_state,
        next_recurrent_state,
        rewards,
        dones,
    ):

        # cleanrl formulation of a DQN-style TD update
        def phi_exp_loss(phi_params):
            phi = phi_net.apply(phi_params, next_obs)
            # normalise_phi_vmap = jax.vmap(normalise_phi)
            # phi = normalise_phi(phi)
            phi = phi / (
                jax.lax.max(jnp.sum(phi**2, -1, keepdims=True), 1e-12) ** (0.5)
            )
            # phi = phi.reshape(traj_len - 1, mb_size, phi_dim)
            phi_loss = jnp.mean(
                1.0 - jnp.sum(jnp.multiply(phi, jax.lax.stop_gradient(w)), axis=-1)
            )
            return phi_loss, phi

        (phi_loss, phi), phi_grads = jax.value_and_grad(phi_exp_loss, has_aux=True)(
            phi_state.params
        )
        phi_state = phi_state.apply_gradients(grads=phi_grads)

        target_psi, _ = psi_net.apply(
            psi_state.target_params, next_obs, w, next_recurrent_state
        )

        next_q = utils.q_from_psi_vmap(target_psi, w)
        next_action = jnp.argmax(next_q, -1)
        target_psi = utils.psi_selector(target_psi, next_action, phi_dim)

        target_psi = (
            phi
            + (1 - jnp.repeat(dones[:, jnp.newaxis], repeats=phi_dim, axis=-1))
            * gamma
            * target_psi
        )

        def mse_loss(psi_params):
            psi, _ = psi_net.apply(
                psi_params, obs, w, recurrent_state
            )  # (batch_size, num_actions)
            psi = psi.reshape(obs.shape[0], -1, phi_dim)
            psi = psi[np.arange(psi.shape[0]), actions.squeeze(), ...]
            body_fun = (
                lambda i, psi_loss: psi_loss
                + ((psi[:, i] - jax.lax.stop_gradient(target_psi[:, i])) ** 2).mean()
            )
            psi_loss = jax.lax.fori_loop(0, phi_dim, body_fun, 0) / phi_dim

            # return ((psi - jax.lax.stop_gradient(target_psi)) ** 2).sum(-1).mean(), psi
            return psi_loss, psi

        (psi_loss, psi), psi_grads = jax.value_and_grad(mse_loss, has_aux=True)(
            psi_state.params
        )
        psi_state = psi_state.apply_gradients(grads=psi_grads)

        return psi_loss, phi_loss, psi_state, phi_state, psi, phi

    # from gymnax example notebooks - modified to be a fixed length rollout
    # for a given length of task

    def rollout(
        obs, env_state, rng_input, w, psi_state, recurrent_state, env_params, traj_len
    ):
        """Rollout a jitted gymnax episode with lax.scan."""
        # Reset the environment
        rng_reset, rng_episode = jax.random.split(rng_input)

        def policy_step(state_input, tmp):
            """lax.scan compatible step transition in jax env."""
            obs, state, policy_params, recurrent_state, rng = state_input
            (
                rng,
                rng_step,
                rng_net,
                env_key,
                recurrent_rng,
                done_key,
                action_key,
            ) = jax.random.split(rng, 7)
            gpi_psi, next_recurrent_state = gpi_act(
                psi_state, obs, w, recurrent_state, rng_net
            )
            random_action = jax.random.randint(
                rng, shape=(1,), minval=0, maxval=num_actions
            )
            action = jax.lax.cond(
                jax.random.uniform(action_key) > epsilon,
                lambda gpi_psi, w: get_gpi_action(gpi_psi, w),
                lambda gpi_psi, w: random_action[0],
                gpi_psi,
                w,
            )

            next_obs, next_state, reward, done, _ = env.step(
                env_key, state, action, env_params
            )
            next_recurrent_state = jax.lax.cond(
                done,
                lambda recurrent_rng, phi_dim: (
                    jnp.zeros_like(next_recurrent_state[0]),
                    jnp.zeros_like(next_recurrent_state[1]),
                ),
                lambda recurrent_rng, phi_dim: next_recurrent_state,
                recurrent_rng,
                phi_dim,
            )

            carry = [
                next_obs,
                next_state,
                policy_params,
                next_recurrent_state,
                rng,
            ]
            return carry, [
                obs,
                action,
                reward,
                next_obs,
                recurrent_state,
                next_recurrent_state,
                done,
            ]

        # Scan over episode step loop
        carry, scan_out = jax.lax.scan(
            policy_step,
            [obs, env_state, psi_state, recurrent_state, rng_episode],
            (),
            traj_len,
        )

        # Return masked sum of rewards accumulated by agent in episode
        (
            obs,
            action,
            reward,
            next_obs,
            recurrent_state,
            next_recurrent_state,
            done,
        ) = scan_out
        state = carry[1]
        return (
            obs,
            state,
            action,
            reward,
            next_obs,
            recurrent_state,
            next_recurrent_state,
            done,
        )

    # single episode run to completion: logic based off of the single
    # rollout function in the gymnax experimental rollout manager class

    def episode(
        obs,
        env_state,
        rng_input,
        w,
        psi_state,
        recurrent_state,
        env_params,
        traj_len,
        eval_episode=True,
    ):
        """Rollout a jitted gymnax episode with lax.scan."""
        # Reset the environment
        rng_reset, rng_episode = jax.random.split(rng_input)

        def policy_step(state_input, tmp):
            """lax.scan compatible step transition in jax env."""
            (
                obs,
                state,
                policy_params,
                recurrent_state,
                rng,
                cum_reward,
                valid_mask,
            ) = state_input
            rng, rng_step, rng_net, env_key, done_key, action_key = jax.random.split(
                rng, 6
            )
            gpi_psi, next_recurrent_state = gpi_act(
                psi_state, obs, w, recurrent_state, rng_net
            )
            random_action = jax.random.randint(
                rng, shape=(1,), minval=0, maxval=num_actions
            )
            if eval_episode:
                action = get_gpi_action(gpi_psi, w)
            else:
                action = jax.lax.cond(
                    jax.random.uniform(action_key) > epsilon,
                    lambda gpi_psi, w: get_gpi_action(gpi_psi, w),
                    lambda gpi_psi, w: random_action[0],
                    gpi_psi,
                    w,
                )

            next_obs, next_state, reward, done, _ = env.step(
                env_key, state, action, env_params
            )

            new_cum_reward = cum_reward + reward * valid_mask
            new_valid_mask = valid_mask * (1 - done)
            carry = [
                next_obs,
                next_state,
                policy_params,
                next_recurrent_state,
                rng,
                new_cum_reward,
                new_valid_mask,
            ]
            return carry, [
                obs,
                action,
                reward,
                next_obs,
                recurrent_state,
                next_recurrent_state,
                done,
            ]

        # Scan over episode step loop
        carry, scan_out = jax.lax.scan(
            policy_step,
            [
                obs,
                env_state,
                psi_state,
                recurrent_state,
                rng_episode,
                jnp.array([0.0]),
                jnp.array([1.0]),
            ],
            (),
            traj_len,
        )

        # Return masked sum of rewards accumulated by agent in episode
        (
            obs,
            action,
            reward,
            next_obs,
            recurrent_state,
            next_recurrent_state,
            done,
        ) = scan_out
        state = carry[1]
        cum_reward = carry[-2]

        return (
            obs,
            state,
            action,
            reward,
            next_obs,
            recurrent_state,
            next_recurrent_state,
            done,
            cum_reward,
        )

    @jax.jit
    def task_inference(next_obs, reward, phi_state):
        next_obs = next_obs.reshape(-1, *obs_shape)
        phi = phi_net.apply(phi_state.params, next_obs)
        phi = phi.reshape(-1, phi_dim)
        phi = phi / (jax.lax.max(jnp.sum(phi**2, -1, keepdims=True), 1e-12) ** (0.5))

        reward = reward.reshape(-1, 1)
        # using solution from https://github.com/google/jax/issues/11433
        lstsq = lambda a, b: jnp.linalg.solve(a.T @ a, a.T @ b)
        w_inferred = lstsq(phi, reward).squeeze()
        w_inferred = w_inferred / jnp.linalg.norm(w_inferred, keepdims=True, ord=2)
        return w_inferred

    jit_rollout = jax.vmap(
        jax.jit(rollout, static_argnums=7), in_axes=[0, 0, 0, 0, None, 0, None, None]
    )

    # don't vmap over w in episode - should be the same
    jit_episode = jax.vmap(
        jax.jit(episode, static_argnums=[7, 8]),
        in_axes=[0, 0, 0, None, None, 0, None, None, None],
    )

    vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    vmap_keys = jax.random.split(rng_key, num_envs)

    # get first observations
    obs, env_state = vmap_reset(vmap_keys, env_params)
    env_steps = 0

    rng_key, lstm_key = jax.random.split(rng_key, 2)
    recurrent_state = nn.OptimizedLSTMCell().initialize_carry(
        lstm_key, batch_dims=(num_envs,), size=lstm_size
    )

    for e in range(num_steps):

        rng_key, _ = jax.random.split(rng_key, 2)
        rng_rollout_batch = jax.random.split(rng_key, num_envs)

        w = utils.sample_uniform_sphere(shape=[num_envs, phi_dim], rng_key=rng_key)

        (
            obs,
            env_state,
            a,
            reward,
            next_obs,
            recurrent_state,
            next_recurrent_state,
            done,
        ) = jit_rollout(
            obs,
            env_state,
            rng_rollout_batch,
            w,
            psi_state,
            recurrent_state,
            env_params,
            traj_len,
        )

        obs_flattened = obs.reshape(-1, *obs_shape)
        next_obs_flattened = next_obs.reshape(-1, *obs_shape)

        a_flattened = a.copy().reshape([-1])

        recurrent_state_flattened = flatten_nested_recurrent_state(recurrent_state)
        next_recurrent_state_flattened = flatten_nested_recurrent_state(
            next_recurrent_state
        )

        done_flattened = done.reshape(-1)
        w_flattened = jnp.repeat(
            w[:, jnp.newaxis, :].copy(), repeats=traj_len, axis=1
        ).reshape(-1, phi_dim)
        psi_loss, phi_loss, psi_state, phi_state, this_psi, this_phi = update(
            psi_state,
            phi_state,
            obs_flattened,
            a_flattened,
            next_obs_flattened,
            w_flattened,
            recurrent_state_flattened,
            next_recurrent_state_flattened,
            None,
            done_flattened,
        )

        env_steps = (e + 1) * num_envs * traj_len

        if e % print_frequency == 0:
            print(
                f"Step {e} ::: Psi Loss {psi_loss} ::: Phi Loss {phi_loss} ::: Env Steps {env_steps}"
            )

        if e % target_update_frequency == 0:
            psi_state = psi_state.replace(
                target_params=optax.incremental_update(
                    psi_state.params, psi_state.target_params, 1
                )
            )

        if e % eval_frequency == 0:
            (
                eval_env_rng,
                eval_sphere_rng,
                eval_rollout_rng,
                rng_key,
                lstm_key,
            ) = jax.random.split(rng_key, 5)
            eval_env_rng_batch = jax.random.split(eval_env_rng, num_eval_envs)
            eval_rollout_rng_batch = jax.random.split(eval_rollout_rng, num_eval_envs)
            eval_recurrent_state = nn.OptimizedLSTMCell().initialize_carry(
                lstm_key, (num_eval_envs,), size=lstm_size
            )
            eval_obs, eval_env_state = vmap_reset(eval_env_rng_batch, env_params)
            w_eval = utils.sample_uniform_sphere(
                shape=[num_eval_envs, phi_dim], rng_key=eval_sphere_rng
            )
            eval_obs, _, _, eval_reward, eval_next_obs, _, _, _ = jit_rollout(
                eval_obs,
                eval_env_state,
                eval_rollout_rng_batch,
                w_eval,
                psi_state,
                eval_recurrent_state,
                env_params,
                num_rewarded_steps // num_eval_envs,
            )

            w_inferred = task_inference(eval_next_obs, eval_reward, phi_state)

            eval_env_rng, eval_sphere_rng, rng_key = jax.random.split(rng_key, 3)
            eval_env_rng_batch = jax.random.split(eval_env_rng, num_eval_envs)

            eval_obs, eval_env_state = vmap_reset(eval_env_rng_batch, env_params)
            eval_recurrent_state = nn.OptimizedLSTMCell().initialize_carry(
                lstm_key, (num_eval_envs,), size=lstm_size
            )

            _, _, _, _, _, _, _, _, cum_reward = jit_episode(
                eval_obs,
                eval_env_state,
                eval_env_rng_batch,
                w_inferred,
                psi_state,
                eval_recurrent_state,
                env_params,
                env_params.max_steps_in_episode,
                True,
            )

            avg_reward = jnp.mean(cum_reward)
            print(f"Average Reward at {env_steps} Steps: {avg_reward}")
            if log_wandb:
                w_dict = {f"w_{i}": w_i for i, w_i in enumerate(np.array(w_inferred))}
                wandb.log(w_dict, commit=False)
                wandb.log({"avg_reward": avg_reward}, commit=False)

        obs = next_obs[:, -1, ...]
        recurrent_state = (
            recurrent_state[0][:, -1, ...],
            recurrent_state[1][:, -1, ...],
        )

        if e % log_frequency == 0:
            if log_wandb:
                wandb.log(
                    {
                        "phi_loss": phi_loss,
                        "psi_loss": psi_loss,
                        "env_steps": env_steps,
                        "updates": e,
                    }
                )

    # save models
    phi_state_dict = flax.serialization.to_state_dict(phi_state)
    psi_state_dict = flax.serialization.to_state_dict(psi_state)

    with open(f"phi_state_{num_steps}_{args.game}_seed{args.seed}.pkl", "wb") as f:
        pickle.dump(phi_state_dict, f)
    with open(f"psi_state_{num_steps}_{args.game}_seed{args.seed}.pkl", "wb") as f:
        pickle.dump(psi_state_dict, f)

    return phi_net, phi_state, psi_net, psi_state


# run loop
# if ( not os.path.exists(f"phi_state_{args.num_steps}_{args.game}.pkl") or args.overwrite):
phi_net, phi_state, psi_net, psi_state = training_loop(args, env, env_params)
