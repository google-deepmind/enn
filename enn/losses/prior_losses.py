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

"""Prior losses are losses that regulate towards the prior."""

import dataclasses

from absl import logging
import chex
from enn import base
from enn import datasets
from enn import networks
from enn import utils
from enn.losses import base as losses_base
import haiku as hk
import jax
import jax.numpy as jnp
import typing_extensions


class FakeInputGenerator(typing_extensions.Protocol):

  def __call__(self, batch: datasets.ArrayBatch,
               key: chex.PRNGKey) -> chex.Array:
    """Generates a fake batch of input=x for use in prior regularization."""


@dataclasses.dataclass
class MatchingGaussianData(FakeInputGenerator):
  """Generates a fake batch of input=x for use in prior regularization."""

  scale: float = 1.

  def __call__(self, batch: datasets.ArrayBatch,
               key: chex.PRNGKey) -> chex.Array:
    """Generates a fake batch of input=x for use in prior regularization."""
    return jax.random.normal(key, batch.x.shape) * self.scale


def make_gaussian_dataset(batch_size: int,
                          input_dim: int,
                          seed: int = 0) -> datasets.ArrayBatchIterator:
  """Returns a batch iterator over random Gaussian data."""
  sample_fn = jax.jit(lambda x: jax.random.normal(x, [batch_size, input_dim]))
  def batch_iterator():
    rng = hk.PRNGSequence(seed)
    while True:
      x = sample_fn(next(rng))
      yield datasets.ArrayBatch(x=x, y=jnp.ones([x.shape[0], 1]))
  return batch_iterator()


def variance_kl(var: chex.Array,
                pred_log_var: chex.Array) -> chex.Array:
  """Compute the KL divergence between Gaussian variance with matched means."""
  log_var = jnp.log(var)
  pred_var = jnp.exp(pred_log_var)
  return 0.5 * (pred_log_var - log_var + var / pred_var - 1)


# TODO(author3): Remove and use generate_batched_forward_at_data_with_state.
def generate_batched_forward_at_data(
    num_index_sample: int, x: chex.Array,
    enn: networks.EnnNoState, params: hk.Params,
    key: chex.PRNGKey) -> networks.Output:
  """Generate enn output for batch of data with indices based on random key."""
  batched_indexer = utils.make_batch_indexer(enn.indexer, num_index_sample)
  batched_forward = jax.vmap(enn.apply, in_axes=[None, None, 0])
  batched_out = batched_forward(params, x, batched_indexer(key))
  return batched_out


def generate_batched_forward_at_data_with_state(
    num_index_sample: int, x: chex.Array,
    enn: networks.EnnArray, params: hk.Params,
    key: chex.PRNGKey) -> networks.Output:
  """Generate enn output for batch of data with indices based on random key."""
  batched_indexer = utils.make_batch_indexer(enn.indexer, num_index_sample)
  batched_forward = jax.vmap(enn.apply, in_axes=[None, None, None, 0])
  unused_state = {}
  batched_out, unused_state = batched_forward(params, unused_state, x,
                                              batched_indexer(key))
  return batched_out


def l2_training_penalty(batched_out: networks.Output):
  """Penalize the L2 magnitude of the training network."""
  if isinstance(batched_out, networks.OutputWithPrior):
    return 0.5 * jnp.mean(jnp.square(batched_out.train))
  else:
    logging.warning('L2 weight penalty only works for OutputWithPrior.')
    return 0.


def distill_mean_regression(
    batched_out: networks.Output,
    distill_out: networks.Output) -> chex.Array:
  """Train the mean of the regression to the distill network."""
  observed_mean = jnp.mean(networks.parse_net_output(batched_out), axis=0)
  distill_mean = jnp.squeeze(networks.parse_net_output(distill_out))
  return jnp.mean(jnp.square(distill_mean - observed_mean))


def distill_mean_classification(
    batched_out: networks.Output,
    distill_out: networks.Output) -> chex.Array:
  """Train the mean of the classification to the distill network."""
  batched_logits = networks.parse_net_output(batched_out)
  batched_probs = jax.nn.softmax(batched_logits, axis=-1)
  mean_probs = jnp.mean(batched_probs, axis=0)
  distill_probs = jax.nn.softmax(
      networks.parse_net_output(distill_out), axis=-1)
  return jnp.mean(
      jnp.sum(mean_probs * jnp.log(mean_probs / distill_probs), axis=1))


def distill_var_regression(batched_out: networks.Output,
                           distill_out: networks.Output) -> chex.Array:
  """Train the variance of the regression to the distill network."""
  assert isinstance(distill_out, networks.OutputWithPrior)
  observed_var = jnp.var(networks.parse_net_output(batched_out), axis=0)
  return jnp.mean(variance_kl(observed_var, distill_out.extra['log_var']))


def distill_var_classification(
    batched_out: networks.Output,
    distill_out: networks.Output) -> chex.Array:
  """Train the variance of the classification to the distill network."""
  assert isinstance(distill_out, networks.OutputWithPrior)
  batched_logits = networks.parse_net_output(batched_out)
  observed_var = jnp.var(jax.nn.softmax(batched_logits, axis=-1))
  return jnp.mean(variance_kl(observed_var, distill_out.extra['log_var']))


@dataclasses.dataclass
class RegressionPriorLoss(losses_base.LossFnArray):
  """Regress fake data back to prior, and distill mean/var to mean_index."""
  num_index_sample: int
  input_generator: FakeInputGenerator = dataclasses.field(
      default_factory=MatchingGaussianData
  )
  scale: float = 1.
  distill_index: bool = False

  def __call__(self, enn: networks.EnnArray,
               params: hk.Params, state: hk.State, batch: datasets.ArrayBatch,
               key: chex.PRNGKey) -> base.LossOutput:
    index_key, data_key = jax.random.split(key)
    fake_x = self.input_generator(batch, data_key)
    # TODO(author2): Complete prior loss refactor --> MultilossExperiment
    batched_out = generate_batched_forward_at_data_with_state(
        self.num_index_sample,
        fake_x,
        enn,
        params,
        index_key,
    )

    # Regularize towards prior output
    loss = self.scale * l2_training_penalty(batched_out)

    # Distill aggregate stats to the "mean_index"
    if hasattr(enn.indexer, 'mean_index') and self.distill_index:
      distill_out = enn.apply(params, fake_x, enn.indexer.mean_index)
      loss += distill_mean_regression(batched_out, distill_out)
      loss += distill_var_regression(batched_out, distill_out)
    return loss, (state, {})  # pytype: disable=bad-return-type  # numpy-scalars


@dataclasses.dataclass
class ClassificationPriorLoss(losses_base.LossFnArray):
  """Penalize fake data back to prior, and distill mean/var to mean_index."""
  num_index_sample: int
  input_generator: FakeInputGenerator = dataclasses.field(
      default_factory=MatchingGaussianData
  )
  scale: float = 1.
  distill_index: bool = False

  def __call__(
      self,
      enn: networks.EnnArray,
      params: hk.Params,
      state: hk.State,
      batch: datasets.ArrayBatch,
      key: chex.PRNGKey,
  ) -> base.LossOutput:
    index_key, data_key = jax.random.split(key)
    fake_x = self.input_generator(batch, data_key)
    # TODO(author2): Complete prior loss refactor --> MultilossExperiment
    batched_out = generate_batched_forward_at_data_with_state(
        self.num_index_sample, fake_x, enn, params, index_key)

    # Regularize towards prior output
    loss = self.scale * l2_training_penalty(batched_out)

    # Distill aggregate stats to the "mean_index"
    if hasattr(enn.indexer, 'mean_index') and self.distill_index:
      distill_out = enn.apply(params, fake_x, enn.indexer.mean_index)
      loss += distill_mean_classification(batched_out, distill_out)
      loss += distill_var_classification(batched_out, distill_out)
    return loss, (state, {})  # pytype: disable=bad-return-type  # numpy-scalars
