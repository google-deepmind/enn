# python3
# pylint: disable=g-bad-file-header
# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Thompson sampling evaluation of ENN agent on GP regression task."""

import functools
from typing import Dict, Optional, Sequence, Tuple

from acme.utils import loggers
from enn import base as enn_base
from enn import utils
from enn.experiments.neurips_2021 import agents
from enn.experiments.neurips_2021 import base as testbed_base
from enn.experiments.neurips_2021 import load
import haiku as hk
import jax
import jax.numpy as jnp
import optax


class ThompsonEnnBandit:
  """Experiment of Thompson sampling bandit."""

  def __init__(
      self,
      enn_config: agents.VanillaEnnConfig,
      input_dim: int,
      noise_std: float,
      num_actions: int,
      steps_per_obs: int = 1,
      kernel_ctor: load.KernelCtor = load.make_benchmark_kernel,
      kernel_ridge: float = 1e-3,
      logger: Optional[loggers.Logger] = None,
      batch_size: int = 8,
      seed: int = 0,
  ):
    """Initialize a Thompson Sampling experiment."""
    self.rng = hk.PRNGSequence(seed)

    # Initializing the agent internals
    prior = testbed_base.PriorKnowledge(
        input_dim=input_dim,
        num_train=1000,
        num_classes=1,
        layers=1,
        noise_std=noise_std,
    )
    self.enn = enn_config.enn_ctor(prior)
    loss_fn = enn_config.loss_ctor(prior, self.enn)
    self._loss = jax.jit(functools.partial(loss_fn, self.enn))
    optimizer = optax.adam(1e-3)

    # Forward network at random index
    def forward(params: hk.Params,
                inputs: enn_base.Array,
                key: enn_base.RngKey) -> enn_base.Array:
      index = self.enn.indexer(key)
      return self.enn.apply(params, inputs, index)
    self._forward = jax.jit(forward)

    # Perform an SGD step on a batch of data
    def sgd_step(
        params: hk.Params,
        opt_state: optax.OptState,
        batch: enn_base.Batch,
        key: enn_base.RngKey
    ) -> Tuple[hk.Params, optax.OptState]:
      grads, _ = jax.grad(self._loss, has_aux=True)(params, batch, key)
      updates, new_opt_state = optimizer.update(grads, opt_state)
      new_params = optax.apply_updates(params, updates)
      return new_params, new_opt_state
    self._sgd_step = jax.jit(sgd_step)

    # Generating the underlying function
    self.actions = jax.random.normal(next(self.rng), [num_actions, input_dim])
    kernel_fn = kernel_ctor(input_dim)
    mean = jnp.zeros(num_actions)
    k_train_train = kernel_fn(self.actions, x2=None, get='nngp')
    k_train_train += kernel_ridge * jnp.eye(num_actions)
    self.mean_rewards = jax.random.multivariate_normal(
        next(self.rng), mean, k_train_train)
    self.max_reward = jnp.max(self.mean_rewards)

    # Initializing the network
    index = self.enn.indexer(next(self.rng))
    self.params = self.enn.init(next(self.rng), self.actions, index)
    self.opt_state = optimizer.init(self.params)
    self._steps_per_obs = steps_per_obs
    self._noise_std = noise_std
    self._batch_size = batch_size
    self.replay = []
    self.logger = (
        logger or loggers.make_default_logger('experiment', time_delta=0))
    self.num_steps = 0
    self.total_regret = 0

    def select_action(params: hk.Params,
                      key: enn_base.RngKey) -> Dict[str, enn_base.Array]:
      net_key, noise_key, _ = jax.random.split(key, 3)
      net_out = forward(params, self.actions, net_key)
      values = utils.parse_net_output(net_out)
      action = jnp.argmax(values, axis=0)
      mean_reward = self.mean_rewards[action]
      reward = mean_reward + jax.random.normal(noise_key)
      regret = self.max_reward - mean_reward
      return {
          'action': action,
          'reward': reward,
          'regret': regret,
      }
    self._select_action = jax.jit(select_action)

    def make_batch(
        replay: Sequence[Dict[str, enn_base.Array]])-> enn_base.Batch:
      x = jnp.vstack([r['x'] for r in replay])
      y = jnp.vstack([r['y'] for r in replay])
      data_index = jnp.vstack([r['data_index'] for r in replay])
      weights = jnp.ones_like(y)
      return enn_base.Batch(x, y, data_index, weights)
    self._make_batch = jax.jit(make_batch)

  def step(self) -> float:
    """Select action, update replay and return the regret."""
    results = self._select_action(self.params, next(self.rng))
    self.replay.append({
        'x': self.actions[results['action']],
        'y': jnp.ones([1, 1]) * results['reward'],
        'data_index': jnp.ones([1, 1], dtype=jnp.int64) * self.num_steps,
    })
    return float(results['regret'])

  def run(self, num_steps: int, log_freq: int = 1):
    """Run a TS experiment for num_steps."""
    for _ in range(num_steps):
      self.num_steps += 1
      regret = self.step()
      self.total_regret += regret
      if self.num_steps % log_freq == 0:
        self.logger.write({
            'total_regret': self.total_regret,
            't': self.num_steps,
        })
      for _ in range(self._steps_per_obs):
        if self.num_steps <= self._batch_size:
          batch = self._make_batch(self.replay)
        else:
          # Jax becomes much faster when using fixed size batches.
          batch = self._make_batch(self.replay[-self._batch_size:])
        self.params, self.opt_state = self._sgd_step(
            self.params, self.opt_state, batch, next(self.rng))


