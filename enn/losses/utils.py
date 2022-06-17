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

from enn import base_legacy
from enn.losses.single_index import SingleIndexLossFn
from enn.losses.single_index_with_state import SingleIndexLossFnWithStateBase
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


# TODO(author3): Remove this module and use the one with state instead.
def add_l2_weight_decay(
    loss_fn: base_legacy.LossFn,
    scale: Union[float, Callable[[hk.Params], hk.Params]],
    predicate: Optional[PredicateFn] = None
) -> base_legacy.LossFn:
  """Adds scale * l2 weight decay to an existing loss function."""
  try:  # Scale is numeric.
    scale = jnp.sqrt(scale)
    scale_fn = lambda ps: jax.tree_map(lambda p: scale * p, ps)
  except TypeError:
    scale_fn = scale  # Assuming scale is a Callable.

  def new_loss(enn: base_legacy.EpistemicNetwork,
               params: hk.Params, batch: base_legacy.Batch,
               key: base_legacy.RngKey) -> base_legacy.Array:
    loss, metrics = loss_fn(enn, params, batch, key)
    decay = l2_weights_with_predicate(scale_fn(params), predicate)
    total_loss = loss +  decay
    metrics['decay'] = decay
    metrics['raw_loss'] = loss
    return total_loss, metrics
  return new_loss


def add_l2_weight_decay_with_state(
    loss_fn: base_legacy.LossFnWithStateBase[base_legacy.Input,
                                             base_legacy.Data],
    scale: Union[float, Callable[[hk.Params], hk.Params]],
    predicate: Optional[PredicateFn] = None
) -> base_legacy.LossFnWithStateBase[base_legacy.Input, base_legacy.Data]:
  """Adds scale * l2 weight decay to an existing loss function."""
  try:  # Scale is numeric.
    scale = jnp.sqrt(scale)
    scale_fn = lambda ps: jax.tree_map(lambda p: scale * p, ps)
  except TypeError:
    scale_fn = scale  # Assuming scale is a Callable.

  def new_loss(
      enn: base_legacy.EpistemicNetworkWithStateBase[base_legacy.Input],
      params: hk.Params, state: hk.State, batch: base_legacy.Data,
      key: base_legacy.RngKey) -> base_legacy.LossOutputWithState:
    loss, (state, metrics) = loss_fn(enn, params, state, batch, key)
    decay = l2_weights_with_predicate(scale_fn(params), predicate)
    total_loss = loss +  decay
    metrics['decay'] = decay
    metrics['raw_loss'] = loss
    return total_loss, (state, metrics)
  return new_loss


# TODO(author3): Remove this module and use the one with state instead.
def combine_single_index_losses_as_metric(
    train_loss: SingleIndexLossFn,
    extra_losses: Dict[str, SingleIndexLossFn],
) -> SingleIndexLossFn:
  """Combines train_loss for training with extra_losses in metrics."""

  def combined_loss(apply: base_legacy.ApplyFn,
                    params: hk.Params, batch: base_legacy.Batch,
                    index: base_legacy.Index) -> base_legacy.LossOutput:
    loss, metrics = train_loss(apply, params, batch, index)
    for name, loss_fn in extra_losses.items():
      extra_loss, extra_metrics = loss_fn(apply, params, batch, index)
      metrics[f'{name}:loss'] = extra_loss
      for key, value in extra_metrics.items():
        metrics[f'{name}:{key}'] = value
    return loss, metrics

  return combined_loss


def combine_single_index_losses_with_state_as_metric(
    train_loss: SingleIndexLossFnWithStateBase[base_legacy.Input,
                                               base_legacy.Data],
    extra_losses: Dict[str, SingleIndexLossFnWithStateBase[base_legacy.Input,
                                                           base_legacy.Data]],
) -> SingleIndexLossFnWithStateBase[base_legacy.Input, base_legacy.Data]:
  """Combines train_loss for training with extra_losses in metrics."""

  def combined_loss(
      apply: base_legacy.ApplyFnWithStateBase[base_legacy.Input],
      params: hk.Params, state: hk.State, batch: base_legacy.Data,
      index: base_legacy.Index) -> base_legacy.LossOutputWithState:
    loss, (state, metrics) = train_loss(apply, params, state, batch, index)
    for name, loss_fn in extra_losses.items():
      extra_loss, (unused_state,
                   extra_metrics) = loss_fn(apply, params, state, batch, index)
      metrics[f'{name}:loss'] = extra_loss
      for key, value in extra_metrics.items():
        metrics[f'{name}:{key}'] = value
    return loss, (state, metrics)

  return combined_loss


# TODO(author3): Remove this module and use the one with state instead.
def combine_losses_as_metric(
    train_loss: base_legacy.LossFn,
    extra_losses: Dict[str, base_legacy.LossFn],
) -> base_legacy.LossFn:
  """Combines train_loss for training with extra_losses in metrics."""

  def combined_loss(enn: base_legacy.EpistemicNetwork,
                    params: hk.Params, batch: base_legacy.Batch,
                    key: base_legacy.RngKey) -> base_legacy.LossOutput:
    loss, metrics = train_loss(enn, params, batch, key)
    for name, loss_fn in extra_losses.items():
      extra_loss, extra_metrics = loss_fn(enn, params, batch, key)
      metrics[f'{name}:loss'] = extra_loss
      for key, value in extra_metrics.items():
        metrics[f'{name}:{key}'] = value
    return loss, metrics

  return combined_loss


def combine_losses_with_state_as_metric(
    train_loss: base_legacy.LossFnWithStateBase[base_legacy.Input,
                                                base_legacy.Data],
    extra_losses: Dict[str, base_legacy.LossFnWithStateBase[base_legacy.Input,
                                                            base_legacy.Data]],
) -> base_legacy.LossFnWithStateBase[base_legacy.Input, base_legacy.Data]:
  """Combines train_loss for training with extra_losses in metrics."""

  def combined_loss(
      enn: base_legacy.EpistemicNetworkWithStateBase[base_legacy.Input],
      params: hk.Params, state: hk.State, batch: base_legacy.Data,
      key: base_legacy.RngKey) -> base_legacy.LossOutputWithState:
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
class CombineLossConfig:
  loss_fn: base_legacy.LossFn
  name: str = 'unnamed'  # Name for the loss function
  weight: float = 1.  # Weight to scale the loss by


def combine_losses(
    losses: Sequence[Union[CombineLossConfig,
                           base_legacy.LossFn]]
) -> base_legacy.LossFn:
  """Combines multiple losses into a single loss."""
  clean_losses: List[CombineLossConfig] = []
  for i, loss in enumerate(losses):
    if not isinstance(loss, CombineLossConfig):
      loss = CombineLossConfig(loss, name=f'loss_{i}')
    clean_losses.append(loss)

  def loss_fn(enn: base_legacy.EpistemicNetwork,
              params: hk.Params, batch: base_legacy.Batch,
              key: base_legacy.RngKey) -> base_legacy.LossOutput:
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


@dataclasses.dataclass
class CombineLossWithStateConfigBase(Generic[base_legacy.Input,
                                             base_legacy.Data]):
  loss_fn: base_legacy.LossFnWithStateBase[base_legacy.Input, base_legacy.Data]
  name: str = 'unnamed'  # Name for the loss function
  weight: float = 1.  # Weight to scale the loss by


# Module specialized to work only with Array inputs and Batch data.
CombineLossWithStateConfig = CombineLossWithStateConfigBase[base_legacy.Array,
                                                            base_legacy.Data]


def combine_losses_with_state(
    losses: Sequence[Union[CombineLossWithStateConfigBase[base_legacy.Input,
                                                          base_legacy.Data],
                           base_legacy.LossFnWithStateBase[base_legacy.Input,
                                                           base_legacy.Data]]]
) -> base_legacy.LossFnWithStateBase[base_legacy.Input, base_legacy.Data]:
  """Combines multiple losses into a single loss."""
  clean_losses: List[CombineLossWithStateConfigBase] = []
  for i, loss in enumerate(losses):
    if not isinstance(loss, CombineLossWithStateConfigBase):
      loss = CombineLossWithStateConfigBase(loss, name=f'loss_{i}')
    clean_losses.append(loss)

  def loss_fn(enn: base_legacy.EpistemicNetworkWithState[base_legacy.Input],
              params: hk.Params, state: hk.State, batch: base_legacy.Data,
              key: base_legacy.RngKey) -> base_legacy.LossOutputWithState:
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
