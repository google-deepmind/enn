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
"""Collection of simple losses applied to one single index."""

import dataclasses
from typing import Callable, Optional

import chex
from enn import base
from enn import data_noise
from enn import networks
from enn import utils
from enn.losses import base as losses_base
import haiku as hk
import jax
import jax.numpy as jnp


def average_single_index_loss(
    single_loss: losses_base.SingleIndexLossFn, num_index_samples: int = 1
) -> losses_base.LossFn:
  """Average a single index loss over multiple index samples.

  Args:
    single_loss: loss function applied per epistemic index.
    num_index_samples: number of index samples to average.

  Returns:
    LossFn that comprises the mean of both the loss and the metrics.
  """

  def loss_fn(enn: networks.EpistemicNetwork,
              params: hk.Params, batch: base.Batch,
              key: chex.PRNGKey) -> losses_base.LossOutput:
    batched_indexer = utils.make_batch_indexer(enn.indexer, num_index_samples)
    batched_loss = jax.vmap(single_loss, in_axes=[None, None, None, 0])
    loss, metrics = batched_loss(enn.apply, params, batch, batched_indexer(key))
    batch_mean = lambda x: jnp.mean(x, axis=0)
    return batch_mean(loss), jax.tree_map(batch_mean, metrics)
  return loss_fn


def add_data_noise(
    single_loss: losses_base.SingleIndexLossFn,
    noise_fn: data_noise.DataNoise,
) -> losses_base.SingleIndexLossFn:
  """Applies a DataNoise function to each batch of data."""

  def noisy_loss(apply: networks.ApplyFn,
                 params: hk.Params, batch: base.Batch,
                 index: base.Index) -> losses_base.LossOutput:
    noisy_batch = noise_fn(batch, index)
    return single_loss(apply, params, noisy_batch, index)
  return noisy_loss


@dataclasses.dataclass
class L2Loss(losses_base.SingleIndexLossFn):
  """L2 regression applied to a single epistemic index."""

  def __call__(self, apply: networks.ApplyFn, params: hk.Params,
               batch: base.Batch,
               index: base.Index) -> losses_base.LossOutput:
    """L2 regression applied to a single epistemic index."""
    chex.assert_shape(batch.y, (None, 1))
    chex.assert_shape(batch.data_index, (None, 1))
    net_out = networks.parse_net_output(apply(params, batch.x, index))
    chex.assert_equal_shape([net_out, batch.y])
    sq_loss = jnp.square(networks.parse_net_output(net_out) - batch.y)
    if batch.weights is None:
      batch_weights = jnp.ones_like(batch.data_index)
    else:
      batch_weights = batch.weights
    chex.assert_equal_shape([batch_weights, sq_loss])
    return jnp.mean(batch_weights * sq_loss), {}


@dataclasses.dataclass
class XentLoss(losses_base.SingleIndexLossFn):
  """Cross-entropy classification single index loss."""
  num_classes: int

  def __post_init__(self):
    chex.assert_scalar_non_negative(self.num_classes - 2.0)

  def __call__(self, apply: networks.ApplyFn, params: hk.Params,
               batch: base.Batch,
               index: base.Index) -> losses_base.LossOutput:
    chex.assert_shape(batch.y, (None, 1))
    chex.assert_shape(batch.data_index, (None, 1))
    net_out = apply(params, batch.x, index)
    logits = networks.parse_net_output(net_out)
    labels = jax.nn.one_hot(batch.y[:, 0], self.num_classes)

    softmax_xent = -jnp.sum(
        labels * jax.nn.log_softmax(logits), axis=1, keepdims=True)
    if batch.weights is None:
      batch_weights = jnp.ones_like(batch.data_index)
    else:
      batch_weights = batch.weights
    chex.assert_equal_shape([batch_weights, softmax_xent])
    return jnp.mean(batch_weights * softmax_xent), {}


@dataclasses.dataclass
class AccuracyErrorLoss(losses_base.SingleIndexLossFn):
  """Evaluates the accuracy error of a greedy logit predictor."""
  num_classes: int

  def __call__(self, apply: networks.ApplyFn, params: hk.Params,
               batch: base.Batch,
               index: base.Index) -> losses_base.LossOutput:
    chex.assert_shape(batch.y, (None, 1))
    net_out = apply(params, batch.x, index)
    logits = networks.parse_net_output(net_out)
    preds = jnp.argmax(logits, axis=1)
    correct = (preds == batch.y[:, 0])
    accuracy = jnp.mean(correct)
    return 1 - accuracy, {'accuracy': accuracy}


@dataclasses.dataclass
class ElboLoss(losses_base.SingleIndexLossFn):
  """Standard VI loss (negative of evidence lower bound).

  Given latent variable u with model density q(u), prior density p_0(u)
  and likelihood function p(D|u) the evidence lower bound is defined as
      ELBO(q) = E[log(p(D|u))] - KL(q(u)||p_0(u))
  In other words, maximizing ELBO is equivalent to regularized log likelihood
  maximization where regularization is encouraging the learned latent
  distribution to be close to the latent prior as measured by KL.
  """

  log_likelihood_fn: Callable[[base.Output, base.Batch], float]
  model_prior_kl_fn: Callable[
      [base.Output, hk.Params, base.Index], float]
  temperature: Optional[float] = None
  input_dim: Optional[int] = None

  def __call__(self, apply: networks.ApplyFn, params: hk.Params,
               batch: base.Batch,
               index: base.Index) -> losses_base.LossOutput:
    """This function returns a one-sample MC estimate of the ELBO."""
    out = apply(params, batch.x, index)
    log_likelihood = self.log_likelihood_fn(out, batch)
    model_prior_kl = self.model_prior_kl_fn(out, params, index)
    chex.assert_equal_shape([log_likelihood, model_prior_kl])
    if self.temperature and self.input_dim:
      model_prior_kl *= jnp.sqrt(self.temperature) * self.input_dim
    return model_prior_kl - log_likelihood, {}


@dataclasses.dataclass
class VaeLoss(losses_base.SingleIndexLossFn):
  """VAE loss."""
  log_likelihood_fn: Callable[[base.OutputWithPrior, base.Batch],
                              float]
  latent_kl_fn: Callable[[base.OutputWithPrior], float]

  def __call__(self, apply: networks.ApplyFn, params: hk.Params,
               batch: base.Batch,
               index: base.Index) -> losses_base.LossOutput:
    net_out = apply(params, batch.x, index)
    kl_term = self.latent_kl_fn(net_out)
    log_likelihood = self.log_likelihood_fn(net_out, batch)
    return kl_term - log_likelihood, {}
