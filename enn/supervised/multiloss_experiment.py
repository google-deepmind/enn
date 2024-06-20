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

"""An SGD experiment with facility for multiple losses."""

import dataclasses
import functools
from typing import Callable, Dict, NamedTuple, Optional, Sequence, Tuple

import chex
from enn import base
from enn import datasets
from enn import loggers
from enn import losses
from enn import networks
from enn.supervised import base as supervised_base
import haiku as hk
import jax
import optax


class TrainingState(NamedTuple):
  params: hk.Params
  network_state: hk.State
  opt_state: optax.OptState


@dataclasses.dataclass
class MultilossTrainer:
  """Specify the training schedule for a given loss/dataset.

  For step=1,2,...:
    If should_train(step):
      Apply one step of loss_fn on a batch = next(dataset).
  """
  loss_fn: losses.LossFnArray  # Loss function
  dataset: datasets.ArrayBatchIterator  # Dataset to pull batch from
  should_train: Callable[[int], bool] = lambda _: True  # Which steps to train
  name: str = 'loss'  # Name used for logging


# Type definition for loss function after internalizing the ENN
PureLoss = Callable[
    [hk.Params, hk.State, datasets.ArrayBatch, chex.PRNGKey],
    base.LossOutput]


class MultilossExperiment(supervised_base.BaseExperiment):
  """Class to handle supervised training with multiple losses.

  At each step=1,2,...:
    For t in trainers:
      If t.should_train(step):
        Apply one step of t.loss_fn on batch = next(t.dataset)

  This can be useful for settings like "prior_loss" or transfer learning.

  Optional eval_datasets which is a collection of datasets to *evaluate*
  the loss on every eval_log_freq steps.
  """

  def __init__(
      self,
      enn: networks.EnnArray,
      trainers: Sequence[MultilossTrainer],
      optimizer: optax.GradientTransformation,
      seed: int = 0,
      logger: Optional[loggers.Logger] = None,
      train_log_freq: int = 1,
      eval_datasets: Optional[Dict[str, datasets.ArrayBatchIterator]] = None,
      eval_log_freq: int = 1,
  ):
    self.enn = enn
    self.pure_trainers = _purify_trainers(trainers, enn)
    self.rng = hk.PRNGSequence(seed)

    # Internalize the eval datasets
    self._eval_datasets = eval_datasets
    self._eval_log_freq = eval_log_freq

    # Forward network at random index
    def forward(params: hk.Params, state: hk.State, inputs: chex.Array,
                key: chex.PRNGKey) -> chex.Array:
      index = self.enn.indexer(key)
      out, _ = self.enn.apply(params, state, inputs, index)
      return out
    self._forward = jax.jit(forward)

    # Define the SGD step on the loss
    def sgd_step(
        pure_loss: PureLoss,
        state: TrainingState,
        batch: datasets.ArrayBatch,
        key: chex.PRNGKey,
    ) -> Tuple[TrainingState, base.LossMetrics]:
      # Calculate the loss, metrics and gradients
      loss_output, grads = jax.value_and_grad(pure_loss, has_aux=True)(
          state.params, state.network_state, batch, key)
      loss, (network_state, metrics) = loss_output
      metrics.update({'loss': loss})
      updates, new_opt_state = optimizer.update(
          updates=grads, state=state.opt_state, params=state.params
      )
      new_params = optax.apply_updates(state.params, updates)
      new_state = TrainingState(
          params=new_params,
          network_state=network_state,
          opt_state=new_opt_state,
      )
      return new_state, metrics

    self._sgd_step = jax.jit(sgd_step, static_argnums=0)

    # Initialize networks
    batch = next(self.pure_trainers[0].dataset)
    index = self.enn.indexer(next(self.rng))
    params, network_state = self.enn.init(next(self.rng), batch.x, index)
    opt_state = optimizer.init(params)
    self.state = TrainingState(params, network_state, opt_state)
    self.step = 0
    self.logger = logger or loggers.make_default_logger(
        'experiment', time_delta=0)
    self._train_log_freq = train_log_freq

  def train(self, num_batches: int):
    """Train the ENN for num_batches."""
    for _ in range(num_batches):
      self.step += 1
      for t in self.pure_trainers:
        if t.should_train(self.step):
          self.state, loss_metrics = self._sgd_step(
              t.pure_loss, self.state, next(t.dataset), next(self.rng))

          # Periodically log this performance as dataset=train.
          if self.step % self._train_log_freq == 0:
            loss_metrics.update({
                'dataset': 'train',
                'step': self.step,
                'sgd': True,
                'trainer': t.name,
            })
            self.logger.write(loss_metrics)

      # Periodically evaluate the other datasets.
      if self._eval_datasets and self.step % self._eval_log_freq == 0:
        for name, dataset in self._eval_datasets.items():
          for t in self.pure_trainers:
            loss, (unused_network_state, metrics) = t.pure_loss(
                self.state.params,
                self.state.network_state,
                next(dataset),
                next(self.rng),
            )
            metrics.update({
                'dataset': name,
                'step': self.step,
                'sgd': False,
                'loss': loss,
                'trainer': t.name,
            })
            self.logger.write(metrics)

  def predict(self, inputs: chex.Array,
              key: chex.PRNGKey) -> chex.Array:
    """Evaluate the trained model at given inputs."""
    return self._forward(
        self.state.params,
        self.state.network_state,
        inputs,
        key,
    )

  def loss(self, batch: datasets.ArrayBatch,
           key: chex.PRNGKey) -> chex.Array:
    """Evaluate the first loss for one batch of data."""
    pure_loss = self.pure_trainers[0].pure_loss
    loss, _ = pure_loss(self.state.params, self.state.network_state, batch, key)
    return loss


@dataclasses.dataclass
class _PureTrainer:
  """An intermediate representation of MultilossTrainer with pure loss."""
  pure_loss: PureLoss  # Pure loss function after internalizing enn
  dataset: datasets.ArrayBatchIterator  # Dataset to pull batch from
  should_train: Callable[[int], bool]  # Whether should train on step
  name: str = 'loss'  # Name used for logging


def _purify_trainers(
    trainers: Sequence[MultilossTrainer],
    enn: networks.EnnArray) -> Sequence[_PureTrainer]:
  """Converts MultilossTrainer to have *pure* loss function including enn."""
  pure_trainers = []
  for t in trainers:
    pure_trainer = _PureTrainer(
        pure_loss=jax.jit(functools.partial(t.loss_fn, enn)),
        dataset=t.dataset,
        should_train=t.should_train,
        name=t.name,
    )
    pure_trainers.append(pure_trainer)
  return tuple(pure_trainers)
