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

"""An standard experiment operating by SGD."""

import functools
from typing import Dict, NamedTuple, Optional, Tuple

from acme.utils import loggers
import chex
from enn import base
from enn import losses
from enn import metrics
from enn import networks
from enn.supervised import base as supervised_base
import haiku as hk
import jax
import optax


class TrainingState(NamedTuple):
  params: hk.Params
  network_state: hk.State
  opt_state: optax.OptState


class Experiment(supervised_base.BaseExperiment):
  """Class to handle supervised training.

  Optional eval_datasets which is a collection of datasets to *evaluate*
  the loss on every eval_log_freq steps. Note that this evaluation assumes
  that the dataset will only be for *one* batch. This means that, if you want
  to evaluate on the whole test set, you should make that batch size the size
  of the entire test set, and that it is *repeated* iterator, so you can sample
  from it multiple times without reaching end of iterator.
  """

  def __init__(self,
               enn: networks.EnnArray,
               loss_fn: losses.LossFnArray,
               optimizer: optax.GradientTransformation,
               dataset: base.BatchIterator,
               seed: int = 0,
               logger: Optional[loggers.Logger] = None,
               train_log_freq: int = 1,
               eval_datasets: Optional[Dict[str, base.BatchIterator]] = None,
               eval_metrics: Optional[Dict[str, metrics.MetricCalculator]] = None,  # pylint:disable=line-too-long
               eval_enn_samples: int = 100,
               eval_log_freq: int = 1,
               init_x: Optional[chex.Array] = None):
    """Initializes an SGD experiment.

    Args:
      enn: ENN mapping arrays to any output.
      loss_fn: Defines the loss for the ENN on a batch of data.
      optimizer: optax optimizer.
      dataset: iterator that produces a training batch.
      seed: initializes random seed from jax.
      logger: optional logger, defaults to acme logger.
      train_log_freq: train logging frequency.
      eval_datasets: Optional dict of extra datasets to evaluate on. Note that
        these evaluate on *one* batch, so should be of appropriate batch size.
      eval_metrics: Optional dict of extra metrics that should be evaluated on
        the eval_datasets.
      eval_enn_samples: number of ENN samples to use in eval_metrics evaluation.
      eval_log_freq: evaluation log frequency.
      init_x: optional input array used to initialize networks. Default none
        works by taking from the training dataset.
    """
    self.enn = enn
    self.dataset = dataset
    self.rng = hk.PRNGSequence(seed)

    # Internalize the loss_fn
    self._loss = jax.jit(functools.partial(loss_fn, self.enn))

    # Internalize the eval datasets and metrics
    self._eval_datasets = eval_datasets
    self._eval_metrics = eval_metrics
    self._eval_log_freq = eval_log_freq
    self._eval_enn_samples = eval_enn_samples
    self._should_eval = True if eval_metrics and eval_datasets else False

    # Forward network at random index
    def forward(params: hk.Params,
                state: hk.State,
                inputs: chex.Array,
                key: chex.PRNGKey) -> chex.Array:
      index = self.enn.indexer(key)
      out, unused_state = self.enn.apply(params, state, inputs, index)
      return out
    self._forward = jax.jit(forward)

    # Batched forward at multiple random indices
    self._batch_fwd = jax.vmap(forward, in_axes=[None, None, None, 0])

    # Define the SGD step on the loss
    def sgd_step(
        training_state: TrainingState,
        batch: base.Batch,
        key: chex.PRNGKey,
    ) -> Tuple[TrainingState, base.LossMetrics]:
      # Calculate the loss, metrics and gradients
      loss_output, grads = jax.value_and_grad(self._loss, has_aux=True)(
          training_state.params, training_state.network_state, batch, key)
      loss, (network_state, loss_metrics) = loss_output
      loss_metrics.update({'loss': loss})
      updates, new_opt_state = optimizer.update(grads, training_state.opt_state)
      new_params = optax.apply_updates(training_state.params, updates)
      new_state = TrainingState(
          params=new_params,
          network_state=network_state,
          opt_state=new_opt_state,
      )
      return new_state, loss_metrics
    self._sgd_step = jax.jit(sgd_step)

    # Initialize networks
    if init_x is None:
      batch = next(self.dataset)
      init_x = batch.x
    index = self.enn.indexer(next(self.rng))
    params, network_state = self.enn.init(next(self.rng), init_x, index)
    opt_state = optimizer.init(params)
    self.state = TrainingState(params, network_state, opt_state)
    self.step = 0
    self.logger = logger or loggers.make_default_logger(
        'experiment', time_delta=0)
    self._train_log_freq = train_log_freq

  def train(self, num_batches: int):
    """Trains the experiment for specified number of batches.

    Note that this training is *stateful*, the experiment keeps track of the
    total number of training steps that have occured. This method *also* logs
    the training and evaluation metrics periodically.

    Args:
      num_batches: the number of training batches, and SGD steps, to perform.
    """
    for _ in range(num_batches):
      self.step += 1
      self.state, loss_metrics = self._sgd_step(
          self.state, next(self.dataset), next(self.rng))

      # Periodically log this performance as dataset=train.
      if self.step % self._train_log_freq == 0:
        loss_metrics.update(
            {'dataset': 'train', 'step': self.step, 'sgd': True})
        self.logger.write(loss_metrics)

      # Periodically evaluate the other datasets.
      if self._should_eval and self.step % self._eval_log_freq == 0:
        for name, dataset in self._eval_datasets.items():
          # Evaluation happens on a single batch
          eval_batch = next(dataset)
          eval_metrics = {'dataset': name, 'step': self.step, 'sgd': False}
          # Forward the network once, then evaluate all the metrics
          logits = self._batch_fwd(
              self.state.params,
              self.state.network_state,
              eval_batch.x,
              jax.random.split(next(self.rng), self._eval_enn_samples),
          )
          for metric_name, metric_calc in self._eval_metrics.items():
            loss_metrics.update({
                metric_name: metric_calc(logits, eval_batch.y),
            })

          # Write all the metrics to the logger
          self.logger.write(eval_metrics)

  def predict(self, inputs: chex.Array, key: chex.PRNGKey) -> chex.Array:
    """Evaluate the trained model at given inputs."""
    return self._forward(
        self.state.params,
        self.state.network_state,
        inputs,
        key,
    )

  def loss(self, batch: base.Batch, key: chex.PRNGKey) -> chex.Array:
    """Evaluate the loss for one batch of data."""
    loss, (unused_network_state, unused_metrics) = self._loss(
        self.state.params,
        self.state.network_state,
        batch,
        key,
    )
    return loss
