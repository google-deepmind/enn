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
"""Single index loss functions *with state* (e.g. BatchNorm)."""

from typing import Callable, Tuple

import chex
from enn import base
from enn import utils
import haiku as hk
import jax
import jax.numpy as jnp
import typing_extensions


class SingleIndexLossFnWithState(typing_extensions.Protocol):
  """Calculates a loss based on one batch of data per index.

  You can use utils.average_single_index_loss to make a LossFnWithState out of
  the SingleIndexLossFnWithState.
  """

  def __call__(
      self,
      apply: base.ApplyFn,
      params: hk.Params,
      state: hk.State,
      batch: base.Batch,
      index: base.Index,
  ) -> Tuple[base.Array, Tuple[hk.State, base.LossMetrics]]:
    """Computes a loss based on one batch of data and one index."""


def average_single_index_loss_with_state(
    single_loss: SingleIndexLossFnWithState,
    num_index_samples: int = 1,
) -> base.LossFnWithState:
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

  def loss_fn(
      enn: base.EpistemicNetworkWithState,
      params: hk.Params,
      state: hk.State,
      batch: base.Batch,
      key: base.RngKey) -> Tuple[base.Array, Tuple[hk.State, base.LossMetrics]]:
    batched_indexer = utils.make_batch_indexer(enn.indexer, num_index_samples)
    batched_loss = jax.vmap(single_loss, in_axes=[None, None, None, None, 0])
    loss, (new_state, metrics) = batched_loss(
        enn.apply, params, state, batch, batched_indexer(key))
    mean_loss = jnp.mean(loss)

    # Average the new state across indices, check the output shapes match.
    mean_new_state = jax.tree_map(lambda s: jnp.mean(s, axis=0), new_state)
    jax.tree_multimap(
        lambda x, y: chex.assert_equal_shape([x, y]), mean_new_state, state)

    mean_metrics = jax.tree_map(jnp.mean, metrics)

    # TODO(author2): Adding a logging method for keeping track of state counter.
    if len(mean_new_state) > 0:  # pylint:disable=g-explicit-length-test
      first_state_layer = mean_new_state[list(mean_new_state.keys())[0]]
      mean_metrics['state_counter'] = jnp.mean(first_state_layer['counter'])
    return mean_loss, (mean_new_state, mean_metrics)
  return loss_fn


class XentLossWithState(SingleIndexLossFnWithState):
  """Cross-entropy single index loss with network state as auxiliary."""

  def __init__(self, num_classes: int):
    assert num_classes > 1
    super().__init__()
    self.num_classes = num_classes
    labeller = lambda x: jax.nn.one_hot(x, self.num_classes)
    self._loss = xent_loss_with_state_custom_labels(labeller)

  def __call__(
      self,
      apply: base.ApplyFnWithState,
      params: hk.Params,
      state: hk.State,
      batch: base.Batch,
      index: base.Index,
  ) -> Tuple[base.Array, Tuple[hk.State, base.LossMetrics]]:
    return self._loss(apply, params, state, batch, index)


def xent_loss_with_state_custom_labels(
    labeller: Callable[[chex.Array], chex.Array]) -> SingleIndexLossFnWithState:
  """Factory method to create a loss function with custom labelling."""

  def single_loss(
      apply: base.ApplyFnWithState,
      params: hk.Params,
      state: hk.State,
      batch: base.Batch,
      index: base.Index,
  ) -> Tuple[base.Array, Tuple[hk.State, base.LossMetrics]]:
    """Xent loss with custom labelling."""
    chex.assert_shape(batch.y, (None, 1))
    net_out, state = apply(params, state, batch.x, index)
    logits = utils.parse_net_output(net_out)
    labels = labeller(batch.y[:, 0])

    softmax_xent = -jnp.sum(
        labels * jax.nn.log_softmax(logits), axis=1, keepdims=True)

    loss = jnp.mean(softmax_xent)
    return loss, (state, {'loss': loss})
  return single_loss
