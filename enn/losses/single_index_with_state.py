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
from enn import networks
from enn.losses import base as losses_base
import haiku as hk
import jax
import jax.numpy as jnp


@dataclasses.dataclass
class L2LossWithState(losses_base.SingleLossFnArray):
  """L2 regression applied to a single epistemic index."""

  def __call__(self,
               apply: networks.ApplyArray,
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


class XentLossWithState(losses_base.SingleLossFnArray):
  """Cross-entropy single index loss with network state as auxiliary."""

  def __init__(self, num_classes: int):
    assert num_classes > 1
    super().__init__()
    self.num_classes = num_classes
    labeller = lambda x: jax.nn.one_hot(x, self.num_classes)
    self._loss = xent_loss_with_state_custom_labels(labeller)

  def __call__(
      self,
      apply: networks.ApplyArray,
      params: hk.Params,
      state: hk.State,
      batch: base.Batch,
      index: base.Index,
  ) -> base.LossOutput:
    return self._loss(apply, params, state, batch, index)


def xent_loss_with_state_custom_labels(
    labeller: Callable[[chex.Array], chex.Array]
) -> losses_base.SingleLossFnArray:
  """Factory method to create a loss function with custom labelling."""

  def single_loss(
      apply: networks.ApplyArray,
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
class AccuracyErrorLossWithState(losses_base.SingleLossFnArray):
  """Evaluates the accuracy error of a greedy logit predictor."""
  num_classes: int

  def single_loss(
      self,
      apply: networks.ApplyArray,
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
class ElboLossWithState(losses_base.SingleLossFnArray):
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
      apply: networks.ApplyArray,
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
class VaeLossWithState(losses_base.SingleLossFnArray):
  """VAE loss."""
  log_likelihood_fn: Callable[[base.OutputWithPrior, base.Batch],
                              float]
  latent_kl_fn: Callable[[base.OutputWithPrior], float]

  def __call__(
      self,
      apply: networks.ApplyArray,
      params: hk.Params,
      state: hk.State,
      batch: base.Batch,
      index: base.Index,
  ) -> base.LossOutput:
    net_out, state = apply(params, state, batch.x, index)
    kl_term = self.latent_kl_fn(net_out)
    log_likelihood = self.log_likelihood_fn(net_out, batch)
    return kl_term - log_likelihood, (state, {})

