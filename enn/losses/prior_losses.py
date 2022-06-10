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

"""Prior losses are losses that regulate towards the prior.

These might take the form of weight regularization, or sampling "fake data".
These prior_losses are used in e.g. supervised/prior_experiment.py.
"""

import dataclasses
from absl import logging
from enn import base_legacy
from enn import utils
import haiku as hk
import jax
import jax.numpy as jnp
import typing_extensions


class FakeInputGenerator(typing_extensions.Protocol):

  def __call__(self, batch: base_legacy.Batch,
               key: base_legacy.RngKey) -> base_legacy.Array:
    """Generates a fake batch of input=x for use in prior regularization."""


@dataclasses.dataclass
class MatchingGaussianData(FakeInputGenerator):
  """Generates a fake batch of input=x for use in prior regularization."""

  scale: float = 1.

  def __call__(self, batch: base_legacy.Batch,
               key: base_legacy.RngKey) -> base_legacy.Array:
    """Generates a fake batch of input=x for use in prior regularization."""
    return jax.random.normal(key, batch.x.shape) * self.scale


def make_gaussian_dataset(batch_size: int,
                          input_dim: int,
                          seed: int = 0) -> base_legacy.BatchIterator:
  """Returns a batch iterator over random Gaussian data."""
  sample_fn = jax.jit(lambda x: jax.random.normal(x, [batch_size, input_dim]))
  def batch_iterator():
    rng = hk.PRNGSequence(seed)
    while True:
      x = sample_fn(next(rng))
      yield base_legacy.Batch(x, y=jnp.ones([x.shape[0], 1]))
  return batch_iterator()


def variance_kl(var: base_legacy.Array,
                pred_log_var: base_legacy.Array) -> base_legacy.Array:
  """Compute the KL divergence between Gaussian variance with matched means."""
  log_var = jnp.log(var)
  pred_var = jnp.exp(pred_log_var)
  return 0.5 * (pred_log_var - log_var + var / pred_var - 1)


def generate_batched_forward_at_data(
    num_index_sample: int, x: base_legacy.Array,
    enn: base_legacy.EpistemicNetwork, params: hk.Params,
    key: base_legacy.RngKey) -> base_legacy.Output:
  """Generate enn output for batch of data with indices based on random key."""
  batched_indexer = utils.make_batch_indexer(enn.indexer, num_index_sample)
  batched_forward = jax.vmap(enn.apply, in_axes=[None, None, 0])
  batched_out = batched_forward(params, x, batched_indexer(key))
  return batched_out


def l2_training_penalty(batched_out: base_legacy.Output):
  """Penalize the L2 magnitude of the training network."""
  if isinstance(batched_out, base_legacy.OutputWithPrior):
    return 0.5 * jnp.mean(jnp.square(batched_out.train))
  else:
    logging.warning('L2 weight penalty only works for OutputWithPrior.')
    return 0.


def distill_mean_regression(
    batched_out: base_legacy.Output,
    distill_out: base_legacy.Output) -> base_legacy.Array:
  """Train the mean of the regression to the distill network."""
  observed_mean = jnp.mean(utils.parse_net_output(batched_out), axis=0)
  distill_mean = jnp.squeeze(utils.parse_net_output(distill_out))
  return jnp.mean(jnp.square(distill_mean - observed_mean))


def distill_mean_classification(
    batched_out: base_legacy.Output,
    distill_out: base_legacy.Output) -> base_legacy.Array:
  """Train the mean of the classification to the distill network."""
  batched_logits = utils.parse_net_output(batched_out)
  batched_probs = jax.nn.softmax(batched_logits, axis=-1)
  mean_probs = jnp.mean(batched_probs, axis=0)
  distill_probs = jax.nn.softmax(utils.parse_net_output(distill_out), axis=-1)
  return jnp.mean(jnp.sum(
      mean_probs * jnp.log(mean_probs / distill_probs), axis=1))


def distill_var_regression(
    batched_out: base_legacy.Output,
    distill_out: base_legacy.Output) -> base_legacy.Array:
  """Train the variance of the regression to the distill network."""
  assert isinstance(distill_out, base_legacy.OutputWithPrior)
  observed_var = jnp.var(utils.parse_net_output(batched_out), axis=0)
  return jnp.mean(variance_kl(observed_var, distill_out.extra['log_var']))


def distill_var_classification(
    batched_out: base_legacy.Output,
    distill_out: base_legacy.Output) -> base_legacy.Array:
  """Train the variance of the classification to the distill network."""
  assert isinstance(distill_out, base_legacy.OutputWithPrior)
  batched_logits = utils.parse_net_output(batched_out)
  observed_var = jnp.var(jax.nn.softmax(batched_logits, axis=-1))
  return jnp.mean(variance_kl(observed_var, distill_out.extra['log_var']))


@dataclasses.dataclass
class RegressionPriorLoss(base_legacy.LossFn):
  """Regress fake data back to prior, and distill mean/var to mean_index."""
  num_index_sample: int
  input_generator: FakeInputGenerator = MatchingGaussianData()
  scale: float = 1.
  distill_index: bool = False

  def __call__(self, enn: base_legacy.EpistemicNetwork, params: hk.Params,
               batch: base_legacy.Batch,
               key: base_legacy.RngKey) -> base_legacy.Array:
    index_key, data_key = jax.random.split(key)
    fake_x = self.input_generator(batch, data_key)
    # TODO(author2): Complete prior loss refactor --> MultilossExperiment
    batched_out = generate_batched_forward_at_data(
        self.num_index_sample, fake_x, enn, params, index_key)

    # Regularize towards prior output
    loss = self.scale * l2_training_penalty(batched_out)

    # Distill aggregate stats to the "mean_index"
    if hasattr(enn.indexer, 'mean_index') and self.distill_index:
      distill_out = enn.apply(params, fake_x, enn.indexer.mean_index)
      loss += distill_mean_regression(batched_out, distill_out)
      loss += distill_var_regression(batched_out, distill_out)
    return loss, {}


@dataclasses.dataclass
class ClassificationPriorLoss(base_legacy.LossFn):
  """Penalize fake data back to prior, and distill mean/var to mean_index."""
  num_index_sample: int
  input_generator: FakeInputGenerator = MatchingGaussianData()
  scale: float = 1.
  distill_index: bool = False

  def __call__(self, enn: base_legacy.EpistemicNetwork, params: hk.Params,
               batch: base_legacy.Batch,
               key: base_legacy.RngKey) -> base_legacy.Array:

    index_key, data_key = jax.random.split(key)
    fake_x = self.input_generator(batch, data_key)
    # TODO(author2): Complete prior loss refactor --> MultilossExperiment
    batched_out = generate_batched_forward_at_data(
        self.num_index_sample, fake_x, enn, params, index_key)

    # Regularize towards prior output
    loss = self.scale * l2_training_penalty(batched_out)

    # Distill aggregate stats to the "mean_index"
    if hasattr(enn.indexer, 'mean_index') and self.distill_index:
      distill_out = enn.apply(params, fake_x, enn.indexer.mean_index)
      loss += distill_mean_classification(batched_out, distill_out)
      loss += distill_var_classification(batched_out, distill_out)
    return loss, {}
