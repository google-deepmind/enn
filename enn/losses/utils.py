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

"""Helpful functions relating to losses."""
import dataclasses
from typing import Any, Callable, Dict, Generic, List, Optional, Sequence, Union

import chex
from enn import base
from enn import data_noise
from enn import networks
from enn.losses import base as losses_base
import haiku as hk
import jax
import jax.numpy as jnp


# Maps Haiku params (module_name, name, value) -> include or not
PredicateFn = Callable[[str, str, Any], bool]


def l2_weights_with_predicate(
    params: hk.Params,
    predicate: Optional[PredicateFn] = None) -> jnp.DeviceArray:
  """Sum of squares of parameter weights that passes predicate_fn."""
  if predicate is not None:
    params = hk.data_structures.filter(predicate, params)
  return sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))


def add_data_noise(
    single_loss: losses_base.SingleLossFn[base.Input, base.Data],
    noise_fn: data_noise.DataNoiseBase[base.Data],
) -> losses_base.SingleLossFn[base.Input, base.Data]:
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


def add_l2_weight_decay(
    loss_fn: base.LossFn[base.Input, base.Data],
    scale: Union[float, Callable[[hk.Params], hk.Params]],
    predicate: Optional[PredicateFn] = None
) -> base.LossFn[base.Input, base.Data]:
  """Adds scale * l2 weight decay to an existing loss function."""
  try:  # Scale is numeric.
    scale = jnp.sqrt(scale)
    scale_fn = lambda ps: jax.tree_map(lambda p: scale * p, ps)
  except TypeError:
    scale_fn = scale  # Assuming scale is a Callable.

  def new_loss(
      enn: base.EpistemicNetwork[base.Input],
      params: hk.Params, state: hk.State, batch: base.Data,
      key: chex.PRNGKey) -> base.LossOutput:
    loss, (state, metrics) = loss_fn(enn, params, state, batch, key)
    decay = l2_weights_with_predicate(scale_fn(params), predicate)
    total_loss = loss +  decay
    metrics['decay'] = decay
    metrics['raw_loss'] = loss
    return total_loss, (state, metrics)
  return new_loss


def combine_single_index_losses_as_metric(
    train_loss: losses_base.SingleLossFn[base.Input, base.Data],
    extra_losses: Dict[str, losses_base.SingleLossFn[base.Input, base.Data]],
) -> losses_base.SingleLossFn[base.Input, base.Data]:
  """Combines train_loss for training with extra_losses in metrics."""

  def combined_loss(
      apply: base.ApplyFn[base.Input],
      params: hk.Params, state: hk.State, batch: base.Data,
      index: base.Index) -> base.LossOutput:
    loss, (state, metrics) = train_loss(apply, params, state, batch, index)
    for name, loss_fn in extra_losses.items():
      extra_loss, (unused_state,
                   extra_metrics) = loss_fn(apply, params, state, batch, index)
      metrics[f'{name}:loss'] = extra_loss
      for key, value in extra_metrics.items():
        metrics[f'{name}:{key}'] = value
    return loss, (state, metrics)

  return combined_loss


def combine_losses_as_metric(
    train_loss: base.LossFn[base.Input, base.Data],
    extra_losses: Dict[str, base.LossFn[base.Input, base.Data]],
) -> base.LossFn[base.Input, base.Data]:
  """Combines train_loss for training with extra_losses in metrics."""

  def combined_loss(
      enn: base.EpistemicNetwork[base.Input],
      params: hk.Params, state: hk.State, batch: base.Data,
      key: chex.PRNGKey) -> base.LossOutput:
    loss, (state, metrics) = train_loss(enn, params, state, batch, key)
    for name, loss_fn in extra_losses.items():
      extra_loss, (unused_state,
                   extra_metrics) = loss_fn(enn, params, state, batch, key)
      metrics[f'{name}:loss'] = extra_loss
      for key, value in extra_metrics.items():
        metrics[f'{name}:{key}'] = value
    return loss, (state, metrics)

  return combined_loss


@dataclasses.dataclass
class CombineLossConfig(Generic[base.Input, base.Data]):
  loss_fn: base.LossFn[base.Input, base.Data]
  name: str = 'unnamed'  # Name for the loss function
  weight: float = 1.  # Weight to scale the loss by


# Module specialized to work only with Array inputs and Batch data.
CombineLossConfigArray = CombineLossConfig[chex.Array, base.Data]


def combine_losses(
    losses: Sequence[Union[CombineLossConfig[base.Input, base.Data],
                           base.LossFn[base.Input, base.Data]]]
) -> base.LossFn[base.Input, base.Data]:
  """Combines multiple losses into a single loss."""
  clean_losses: List[CombineLossConfig] = []
  for i, loss in enumerate(losses):
    if not isinstance(loss, CombineLossConfig):
      loss = CombineLossConfig(loss, name=f'loss_{i}')
    clean_losses.append(loss)

  def loss_fn(enn: base.EpistemicNetwork[base.Input],
              params: hk.Params, state: hk.State, batch: base.Data,
              key: chex.PRNGKey) -> base.LossOutput:
    combined_loss = 0.
    combined_metrics = {}
    for loss_config in clean_losses:
      # Compute the loss types for use in conditional computation
      # TODO(author3): This section is a bit of a hack, since we do not have a
      # clear way to deal with network "state" when we combine multiple losses.
      # For now, we just return the input state, but this is not correct when
      # state is not empty.
      loss, (unused_state,
             metrics) = loss_config.loss_fn(enn, params, state, batch, key)
      combined_metrics[f'{loss_config.name}:loss'] = loss
      for name, value in metrics.items():
        combined_metrics[f'{loss_config.name}:{name}'] = value
      combined_loss += loss_config.weight * loss
    return combined_loss, (state, combined_metrics)

  return loss_fn


# The utility loss functions above assume that the enn has a state.
# Since an enn might not have a state, below we define utility loss functions
# for loss functions without state.
# TODO(author3): Remove these utility fns and use the one with state instead.


def wrap_loss_no_state_as_loss(
    loss_fn: losses_base.LossFnNoState,
    constant_state: Optional[hk.State] = None,
) -> losses_base.LossFnArray:
  """Wraps a legacy enn loss with no state as an enn loss."""
  if constant_state is None:
    constant_state = {}
  def new_loss(
      enn: networks.EnnArray,
      params: hk.Params,
      unused_state: hk.State,
      batch: base.Batch,
      key: chex.PRNGKey
  ) -> base.LossOutput:
    enn = networks.wrap_enn_with_state_as_enn(enn)
    loss, metrics = loss_fn(enn, params, batch, key)
    return loss, (constant_state, metrics)

  return new_loss


def wrap_single_loss_no_state_as_single_loss(
    single_loss: losses_base.SingleLossFnNoState,
    constant_state: Optional[hk.State] = None,
) -> losses_base.SingleLossFnArray:
  """Wraps a legacy enn single loss with no state as an enn single loss."""
  if constant_state is None:
    constant_state = {}
  def new_loss(
      apply: networks.ApplyArray,
      params: hk.Params,
      unused_state: hk.State,
      batch: base.Batch,
      index: base.Index,
  ) -> base.LossOutput:
    def apply_no_state(params: hk.Params, x: chex.Array,
                       z: base.Index) -> base.Output:
      output, unused_state = apply(params, constant_state, x, z)
      return output

    loss, metrics = single_loss(apply_no_state, params, batch, index)
    return loss, (constant_state, metrics)

  return new_loss


def add_data_noise_no_state(
    single_loss: losses_base.SingleLossFnNoState,
    noise_fn: data_noise.DataNoise,
) -> losses_base.SingleLossFnNoState:
  """Applies a DataNoise function to each batch of data."""

  def noisy_loss(apply: networks.ApplyNoState,
                 params: hk.Params, batch: base.Batch,
                 index: base.Index) -> losses_base.LossOutputNoState:
    noisy_batch = noise_fn(batch, index)
    return single_loss(apply, params, noisy_batch, index)
  return noisy_loss


def add_l2_weight_decay_no_state(
    loss_fn: losses_base.LossFnNoState,
    scale: Union[float, Callable[[hk.Params], hk.Params]],
    predicate: Optional[PredicateFn] = None
) -> losses_base.LossFnNoState:
  """Adds scale * l2 weight decay to an existing loss function."""
  try:  # Scale is numeric.
    scale = jnp.sqrt(scale)
    scale_fn = lambda ps: jax.tree_map(lambda p: scale * p, ps)
  except TypeError:
    scale_fn = scale  # Assuming scale is a Callable.

  def new_loss(enn: networks.EnnNoState,
               params: hk.Params, batch: base.Batch,
               key: chex.PRNGKey) -> losses_base.LossOutputNoState:
    loss, metrics = loss_fn(enn, params, batch, key)
    decay = l2_weights_with_predicate(scale_fn(params), predicate)
    total_loss = loss +  decay
    metrics['decay'] = decay
    metrics['raw_loss'] = loss
    return total_loss, metrics
  return new_loss


def combine_single_index_losses_no_state_as_metric(
    train_loss: losses_base.SingleLossFnNoState,
    extra_losses: Dict[str, losses_base.SingleLossFnNoState],
) -> losses_base.SingleLossFnNoState:
  """Combines train_loss for training with extra_losses in metrics."""

  def combined_loss(apply: networks.ApplyNoState,
                    params: hk.Params, batch: base.Batch,
                    index: base.Index) -> losses_base.LossOutputNoState:
    loss, metrics = train_loss(apply, params, batch, index)
    for name, loss_fn in extra_losses.items():
      extra_loss, extra_metrics = loss_fn(apply, params, batch, index)
      metrics[f'{name}:loss'] = extra_loss
      for key, value in extra_metrics.items():
        metrics[f'{name}:{key}'] = value
    return loss, metrics

  return combined_loss


def combine_losses_no_state_as_metric(
    train_loss: losses_base.LossFnNoState,
    extra_losses: Dict[str, losses_base.LossFnNoState],
) -> losses_base.LossFnNoState:
  """Combines train_loss for training with extra_losses in metrics."""

  def combined_loss(enn: networks.EnnNoState,
                    params: hk.Params, batch: base.Batch,
                    key: chex.PRNGKey) -> losses_base.LossOutputNoState:
    loss, metrics = train_loss(enn, params, batch, key)
    for name, loss_fn in extra_losses.items():
      extra_loss, extra_metrics = loss_fn(enn, params, batch, key)
      metrics[f'{name}:loss'] = extra_loss
      for key, value in extra_metrics.items():
        metrics[f'{name}:{key}'] = value
    return loss, metrics

  return combined_loss


@dataclasses.dataclass
class CombineLossConfigNoState:
  loss_fn: losses_base.LossFnNoState
  name: str = 'unnamed'  # Name for the loss function
  weight: float = 1.  # Weight to scale the loss by


def combine_losses_no_state(
    losses: Sequence[Union[CombineLossConfigNoState, losses_base.LossFnNoState]]
) -> losses_base.LossFnNoState:
  """Combines multiple losses into a single loss."""
  clean_losses: List[CombineLossConfigNoState] = []
  for i, loss in enumerate(losses):
    if not isinstance(loss, CombineLossConfigNoState):
      loss = CombineLossConfigNoState(loss, name=f'loss_{i}')
    clean_losses.append(loss)

  def loss_fn(enn: networks.EnnNoState,
              params: hk.Params, batch: base.Batch,
              key: chex.PRNGKey) -> losses_base.LossOutputNoState:
    combined_loss = 0.
    combined_metrics = {}
    for loss_config in clean_losses:
      # Compute the loss types for use in conditional computation
      loss, metrics = loss_config.loss_fn(enn, params, batch, key)
      combined_metrics[f'{loss_config.name}:loss'] = loss
      for name, value in metrics.items():
        combined_metrics[f'{loss_config.name}:{name}'] = value
      combined_loss += loss_config.weight * loss
    return combined_loss, combined_metrics

  return loss_fn
