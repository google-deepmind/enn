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

"""An standard experiment operating by SGD.

Includes functionality for a secondary prior_loss_fn that operates every
prior_loss_freq steps.
"""
import functools
from typing import Callable, Dict, Optional, Tuple, Union

from acme.utils import loggers
from enn import base
from enn import losses
from enn.supervised import base as supervised_base
import haiku as hk
import jax
import optax


class Experiment(supervised_base.BaseExperiment):
  """Class to handle supervised training.

  Includes functionality for a secondary prior_loss_fn that operates every
  prior_loss_freq steps. Also optional eval_datasets which is a collection
  of datasets to *evaluate* the loss on every eval_log_freq steps.
  """

  def __init__(self,
               enn: base.EpistemicNetwork,
               loss_fn: Union[base.LossFn, losses.SingleIndexLossFn],
               optimizer: optax.GradientTransformation,
               dataset: base.BatchIterator,
               seed: int = 0,
               logger: Optional[loggers.Logger] = None,
               train_log_freq: int = 1,
               prior_loss_fn: Optional[base.LossFn] = None,
               prior_loss_freq: int = 1,
               eval_datasets: Optional[Dict[str, base.BatchIterator]] = None,
               eval_log_freq: int = 1):
    self.enn = enn
    self.dataset = dataset
    self.rng = hk.PRNGSequence(seed)

    # Internalize the loss_fn (coerce from single_loss if necessary)
    if isinstance(loss_fn, losses.SingleIndexLossFn):
      print(f'WARNING: You have passed SingleIndexLossFn={loss_fn}.'
            '\nThis is coerced to a LossFn by sampling a single random index.')
      loss_fn = losses.average_single_index_loss(loss_fn, num_index_samples=1)
    self._loss = jax.jit(functools.partial(loss_fn, self.enn))

    # Internalize the prior loss if given
    self._prior_loss_freq = prior_loss_freq
    self._prior_loss = None
    if prior_loss_fn:
      self._prior_loss = jax.jit(functools.partial(prior_loss_fn, self.enn))

    # Internalize the eval datasets
    self._eval_datasets = eval_datasets
    self._eval_log_freq = eval_log_freq

    # Forward network at random index
    def forward(
        params: hk.Params, inputs: base.Array, key: base.RngKey) -> base.Array:
      index = self.enn.indexer(key)
      return self.enn.apply(params, inputs, index)
    self._forward = jax.jit(forward)

    # Define the SGD step on the loss
    def sgd_step(
        pure_loss: Callable[[hk.Params, base.Batch, base.RngKey], base.Array],
        params: hk.Params,
        batch: base.Batch,
        key: base.RngKey,
        opt_state: optax.OptState,
    ) -> Tuple[hk.Params, optax.OptState, base.LossMetrics]:
      # Calculate the loss, metrics and gradients
      (loss, metrics), grads = jax.value_and_grad(pure_loss, has_aux=True)(
          params, batch, key)
      metrics.update({'loss': loss})
      updates, new_opt_state = optimizer.update(grads, opt_state)
      new_params = optax.apply_updates(params, updates)
      return new_params, new_opt_state, metrics
    self._sgd_step = jax.jit(sgd_step, static_argnums=0)

    # Initialize networks
    batch = next(self.dataset)
    index = self.enn.indexer(next(self.rng))
    self.params = self.enn.init(next(self.rng), batch['x'], index)
    self.opt_state = optimizer.init(self.params)
    self.step = 0
    self.logger = logger or loggers.make_default_logger('experiment')
    self._train_log_freq = train_log_freq

  def train(self, num_batches: int):
    """Train the ENN for num_batches."""
    for _ in range(num_batches):
      self.step += 1
      self.params, self.opt_state, loss_metrics = self._sgd_step(
          self._loss, self.params, next(self.dataset), next(self.rng),
          self.opt_state)

      # Sometimes do a prior_sgd_step if that is warranted
      if self._prior_loss and self.step % self._prior_loss_freq == 0:
        self.params, self.opt_state, prior_metrics = self._sgd_step(
            self._prior_loss, self.params, next(self.dataset), next(self.rng),
            self.opt_state)
        loss_metrics['prior_loss'] = prior_metrics['loss']

      # Periodically log this performance as dataset=train.
      if self.step % self._train_log_freq == 0:
        loss_metrics.update(
            {'dataset': 'train', 'step': self.step, 'sgd': True})
        self.logger.write(loss_metrics)

      # Periodically evaluate the other datasets.
      if self._eval_datasets and self.step % self._eval_log_freq == 0:
        for name, dataset in self._eval_datasets.items():
          loss, metrics = self._loss(self.params, next(dataset), next(self.rng))
          metrics.update(
              {'dataset': name, 'step': self.step, 'sgd': False, 'loss': loss})
          self.logger.write(metrics)

  def predict(self, inputs: base.Array, seed: int) -> base.Array:
    """Evaluate the trained model at given inputs."""
    return self._forward(self.params, inputs, jax.random.PRNGKey(seed))

  def loss(self, batch: base.Batch, seed: int) -> base.Array:
    """Evaluate the loss for one batch of data."""
    return self._loss(self.params, batch, jax.random.PRNGKey(seed))
