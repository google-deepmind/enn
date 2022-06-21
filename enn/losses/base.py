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
"""Base for losses."""
from typing import Tuple
import chex
from enn import base
from enn import networks
from enn import utils
import haiku as hk
import jax
import jax.numpy as jnp
import typing_extensions as te


class SingleLossFn(te.Protocol[base.Input, base.Output, base.Data,]):
  """Calculates a loss based on one batch of data per index.

  You can use average_single_index_loss to make a base.LossFn out of the
  SingleLossFn.
  """

  def __call__(
      self,
      apply: base.ApplyFn[base.Input, base.Output],
      params: hk.Params,
      state: hk.State,
      batch: base.Data,
      index: base.Index,
  ) -> base.LossOutput:
    """Computes a loss based on one batch of data and one index."""


def average_single_index_loss(
    single_loss: SingleLossFn[base.Input, base.Output, base.Data],
    num_index_samples: int = 1,
) -> base.LossFn[base.Input, base.Output, base.Data]:
  """Average a single index loss over multiple index samples.

  Note that the *network state* is also averaged over indices. This is not going
  to be equivalent to num_index_samples updates sequentially. We may want to
  think about alternative ways to do this, or set num_index_samples=1.

  Args:
    single_loss: loss function applied per epistemic index.
    num_index_samples: number of index samples to average.

  Returns:
    LossFn that comprises the mean of both the loss and the metrics.
  """

  def loss_fn(enn: base.EpistemicNetwork[base.Input, base.Output],
              params: hk.Params,
              state: hk.State,
              batch: base.Data,
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


# Loss modules specialized to work only with Array inputs and Batch data.
LossFnArray = base.LossFn[chex.Array, networks.Output, base.Batch]
SingleLossFnArray = SingleLossFn[chex.Array, networks.Output, base.Batch]

################################################################################
# The default loss definitions above assume that the enn has a state.
# Since an enn might not have a state, below we provide definitions for
# loss functions which work with networks.EnnNoState, specialized to work with
# Array inputs.

# Defining the type for the output of loss functions without state.
LossOutputNoState = Tuple[chex.Array, base.LossMetrics]


class LossFnNoState(te.Protocol):
  """Calculates a loss based on one batch of data per random key."""

  def __call__(self,
               enn: networks.EnnNoState,
               params: hk.Params,
               batch: base.Batch,
               key: chex.PRNGKey) -> LossOutputNoState:
    """Computes a loss based on one batch of data and a random key."""


class SingleLossFnNoState(te.Protocol):
  """Calculates a loss based on one batch of data per index.

  You can use average_single_index_loss_no_state defined below to make a
  LossFnNoState out of the SingleLossFnNoState.
  """

  def __call__(self,
               apply: networks.ApplyNoState,
               params: hk.Params,
               batch: base.Batch,
               index: base.Index) -> LossOutputNoState:
    """Computes a loss based on one batch of data and one index."""


def average_single_index_loss_no_state(
    single_loss: SingleLossFnNoState,
    num_index_samples: int = 1) -> LossFnNoState:
  """Average a single index loss over multiple index samples.

  Args:
    single_loss: loss function applied per epistemic index.
    num_index_samples: number of index samples to average.

  Returns:
    LossFnNoState that comprises the mean of both the loss and the metrics.
  """

  def loss_fn(enn: networks.EnnNoState,
              params: hk.Params,
              batch: base.Batch,
              key: chex.PRNGKey) -> LossOutputNoState:
    batched_indexer = utils.make_batch_indexer(enn.indexer, num_index_samples)
    batched_loss = jax.vmap(single_loss, in_axes=[None, None, None, 0])
    loss, metrics = batched_loss(enn.apply, params, batch, batched_indexer(key))
    batch_mean = lambda x: jnp.mean(x, axis=0)
    return batch_mean(loss), jax.tree_map(batch_mean, metrics)
  return loss_fn
