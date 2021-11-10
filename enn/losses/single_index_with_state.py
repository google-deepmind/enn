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

import dataclasses
from typing import Tuple

import chex
from enn import base as enn_base
from enn import utils as enn_utils
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
      apply: enn_base.ApplyFn,
      params: hk.Params,
      state: hk.State,
      batch: enn_base.Batch,
      index: enn_base.Index,
  ) -> Tuple[enn_base.Array, enn_base.LossMetrics]:
    """Computes a loss based on one batch of data and one index."""


def average_single_index_loss_with_state(
    single_loss: SingleIndexLossFnWithState,
    num_index_samples: int = 1,
) -> enn_base.LossFnWithState:
  """Average a single index loss over multiple index samples.

  Args:
    single_loss: loss function applied per epistemic index.
    num_index_samples: number of index samples to average.

  Returns:
    LossFnWithState that comprises the mean of both the loss and the metrics.
  """

  def loss_fn(enn: enn_base.EpistemicNetworkWithState, params: hk.Params,
              state: hk.Params, batch: enn_base.Batch,
              key: enn_base.RngKey) -> enn_base.Array:
    batched_indexer = enn_utils.make_batch_indexer(
        enn.indexer, num_index_samples)
    batched_loss = jax.vmap(single_loss, in_axes=[None, None, None, None, 0])
    loss, metrics = batched_loss(
        enn.apply, params, state, batch, batched_indexer(key))
    return jnp.mean(loss), jax.tree_map(jnp.mean, metrics)
  return loss_fn


@dataclasses.dataclass
class XentLossWithState(SingleIndexLossFnWithState):
  """Cross-entropy single index loss with network state as auxiliary."""
  num_classes: int

  def __post_init__(self):
    assert self.num_classes > 1

  def __call__(
      self, apply: enn_base.ApplyFnWithState, params: hk.Params,
      state: hk.State, batch: enn_base.Batch,
      index: enn_base.Index) -> Tuple[enn_base.Array, enn_base.LossMetrics]:

    logits, state = apply(params, state, batch.x, index)
    chex.assert_shape(batch.y, (None, self.num_classes))
    labels = batch.y

    softmax_xent = -jnp.sum(
        labels * jax.nn.log_softmax(logits), axis=1, keepdims=True)

    loss = jnp.mean(softmax_xent)
    return loss, {'loss': loss, 'state': state}
