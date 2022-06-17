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
"""Single index loss functions *with state* (e.g. BatchNorm)."""

# TODO(author3): Rename this file to single_index.py and remove WithState from
# all module names.

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


@dataclasses.dataclass
class L2LossWithState(losses_base.SingleIndexLossFnWithState):
  """L2 regression applied to a single epistemic index."""

  def __call__(self,
               apply: networks.ApplyFnWithState,
               params: hk.Params,
               state: hk.State,
               batch: base.Batch,
               index: base.Index,) -> base.LossOutput:
    """L2 regression applied to a single epistemic index."""
    chex.assert_shape(batch.y, (None, 1))
    chex.assert_shape(batch.data_index, (None, 1))
    net_out, state = apply(params, state, batch.x, index)
    net_out = networks.parse_net_output(net_out)
    chex.assert_equal_shape([net_out, batch.y])
    sq_loss = jnp.square(networks.parse_net_output(net_out) - batch.y)
    if batch.weights is None:
      batch_weights = jnp.ones_like(batch.data_index)
    else:
      batch_weights = batch.weights
    chex.assert_equal_shape([batch_weights, sq_loss])
    return jnp.mean(batch_weights * sq_loss), (state, {})


class XentLossWithState(losses_base.SingleIndexLossFnWithState):
  """Cross-entropy single index loss with network state as auxiliary."""

  def __init__(self, num_classes: int):
    assert num_classes > 1
    super().__init__()
    self.num_classes = num_classes
    labeller = lambda x: jax.nn.one_hot(x, self.num_classes)
    self._loss = xent_loss_with_state_custom_labels(labeller)

  def __call__(
      self,
      apply: networks.ApplyFnWithState,
      params: hk.Params,
      state: hk.State,
      batch: base.Batch,
      index: base.Index,
  ) -> base.LossOutput:
    return self._loss(apply, params, state, batch, index)


def xent_loss_with_state_custom_labels(
    labeller: Callable[[chex.Array], chex.Array]
) -> losses_base.SingleIndexLossFnWithState:
  """Factory method to create a loss function with custom labelling."""

  def single_loss(
      apply: networks.ApplyFnWithState,
      params: hk.Params,
      state: hk.State,
      batch: base.Batch,
      index: base.Index,
  ) -> base.LossOutput:
    """Xent loss with custom labelling."""
    chex.assert_shape(batch.y, (None, 1))
    net_out, state = apply(params, state, batch.x, index)
    logits = networks.parse_net_output(net_out)
    labels = labeller(batch.y[:, 0])

    softmax_xent = -jnp.sum(
        labels * jax.nn.log_softmax(logits), axis=1, keepdims=True)

    if batch.weights is None:
      batch_weights = jnp.ones_like(batch.y)
    else:
      batch_weights = batch.weights
    chex.assert_equal_shape([batch_weights, softmax_xent])

    loss = jnp.mean(batch_weights * softmax_xent)
    return loss, (state, {'loss': loss})
  return single_loss


@dataclasses.dataclass
class AccuracyErrorLossWithState(losses_base.SingleIndexLossFnWithState):
  """Evaluates the accuracy error of a greedy logit predictor."""
  num_classes: int

  def single_loss(
      self,
      apply: networks.ApplyFnWithState,
      params: hk.Params,
      state: hk.State,
      batch: base.Batch,
      index: base.Index,
  ) -> base.LossOutput:
    chex.assert_shape(batch.y, (None, 1))
    net_out, state = apply(params, state, batch.x, index)
    logits = networks.parse_net_output(net_out)
    preds = jnp.argmax(logits, axis=1)
    correct = (preds == batch.y[:, 0])
    accuracy = jnp.mean(correct)
    return 1 - accuracy, (state, {'accuracy': accuracy})


@dataclasses.dataclass
class ElboLossWithState(losses_base.SingleIndexLossFnWithState):
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

  def __call__(
      self,
      apply: networks.ApplyFnWithState,
      params: hk.Params,
      state: hk.State,
      batch: base.Batch,
      index: base.Index,
  ) -> base.LossOutput:
    """This function returns a one-sample MC estimate of the ELBO."""
    out, state = apply(params, state, batch.x, index)
    log_likelihood = self.log_likelihood_fn(out, batch)
    model_prior_kl = self.model_prior_kl_fn(out, params, index)
    chex.assert_equal_shape([log_likelihood, model_prior_kl])
    if self.temperature and self.input_dim:
      model_prior_kl *= jnp.sqrt(self.temperature) * self.input_dim
    return model_prior_kl - log_likelihood, (state, {})


@dataclasses.dataclass
class VaeLossWithState(losses_base.SingleIndexLossFnWithState):
  """VAE loss."""
  log_likelihood_fn: Callable[[base.OutputWithPrior, base.Batch],
                              float]
  latent_kl_fn: Callable[[base.OutputWithPrior], float]

  def __call__(
      self,
      apply: networks.ApplyFnWithState,
      params: hk.Params,
      state: hk.State,
      batch: base.Batch,
      index: base.Index,
  ) -> base.LossOutput:
    net_out, state = apply(params, state, batch.x, index)
    kl_term = self.latent_kl_fn(net_out)
    log_likelihood = self.log_likelihood_fn(net_out, batch)
    return kl_term - log_likelihood, (state, {})


def average_single_index_loss_with_state(
    single_loss: losses_base.SingleIndexLossFnWithStateBase[base.Input,
                                                            base.Data],
    num_index_samples: int = 1,
) -> base.LossFn[base.Input, base.Data]:
  """Average a single index loss over multiple index samples.

  Note that the *network state* is also averaged over indices. This is not going
  to be equivalent to num_index_samples updates sequentially. We may want to
  think about alternative ways to do this, or set num_index_samples=1.

  Args:
    single_loss: loss function applied per epistemic index.
    num_index_samples: number of index samples to average.

  Returns:
    LossFnWithState that comprises the mean of both the loss and the metrics.
  """

  def loss_fn(enn: base.EpistemicNetwork[base.Input],
              params: hk.Params, state: hk.State, batch: base.Data,
              key: chex.PRNGKey) -> base.LossOutput:
    # Apply the loss in parallel over num_index_samples different indices.
    # This is the key logic to this loss function.
    batched_indexer = utils.make_batch_indexer(enn.indexer, num_index_samples)
    batched_loss = jax.vmap(single_loss, in_axes=[None, None, None, None, 0])
    loss, (new_state, metrics) = batched_loss(
        enn.apply, params, state, batch, batched_indexer(key))

    # Take the mean over the synthetic index batch dimension
    batch_mean = lambda x: jnp.mean(x, axis=0)
    mean_loss = batch_mean(loss)

    if new_state:
      # TODO(author2): This section is a bit of a hack, since we do not have
      # a clear way to deal with network "state" in the presence of epistemic
      # index. We choose to average the state across epistemic indices and
      # then perform basic error checking to make sure the shape is unchanged.
      new_state = jax.tree_map(batch_mean, new_state)
      jax.tree_multimap(
          lambda x, y: chex.assert_equal_shape([x, y]), new_state, state)
    mean_metrics = jax.tree_map(batch_mean, metrics)

    # TODO(author2): Adding a logging method for keeping track of state counter.
    # This piece of code is only used for debugging/metrics.
    if len(new_state) > 0:  # pylint:disable=g-explicit-length-test
      first_state_layer = new_state[list(new_state.keys())[0]]
      mean_metrics['state_counter'] = jnp.mean(first_state_layer['counter'])
    return mean_loss, (new_state, mean_metrics)
  return loss_fn


def add_data_noise_to_loss_with_state(
    single_loss: losses_base.SingleIndexLossFnWithStateBase[base.Input,
                                                            base.Data],
    noise_fn: data_noise.DataNoiseBase[base.Data],
) -> losses_base.SingleIndexLossFnWithStateBase[base.Input, base.Data]:
  """Applies a DataNoise function to each batch of data."""

  def noisy_loss(
      apply: base.ApplyFn[base.Input],
      params: hk.Params,
      state: hk.State,
      batch: base.Data,
      index: base.Index,
  ) -> base.LossOutput:
    noisy_batch = noise_fn(batch, index)
    return single_loss(apply, params, state, noisy_batch, index)
  return noisy_loss
