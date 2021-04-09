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

"""Helpful functions relating to losses."""
from typing import Dict, Optional, Tuple

from enn import base
from enn.losses import single_index
import haiku as hk
import jax
import jax.numpy as jnp


def l2_weights_excluding_name(params: hk.Params,
                              exclude: Optional[str] = None) -> jnp.DeviceArray:
  """Sum of squares of parameter weights, but not if exclude in name."""
  if exclude:
    predicate = lambda module_name, name, value: exclude not in module_name
    params = hk.data_structures.filter(predicate, params)
  return sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))


def add_l2_weight_decay(loss_fn: base.LossFn,
                        scale: float,
                        exclude: Optional[str] = None) -> base.LossFn:
  """Adds scale * l2 weight decay to an existing loss function."""
  def new_loss(a, params: hk.Params, c, d) -> base.Array:
    loss, metrics = loss_fn(a, params, c, d)
    decay = l2_weights_excluding_name(params, exclude)
    total_loss = loss + scale * decay
    metrics['decay'] = decay
    metrics['raw_loss'] = loss
    return total_loss, metrics
  return new_loss


def combine_single_index_losses_as_metric(
    train_loss: single_index.SingleIndexLossFn,
    extra_losses: Dict[str, single_index.SingleIndexLossFn],
) -> single_index.SingleIndexLossFn:
  """Combines train_loss for training with extra_losses in metrics."""

  def combined_loss(apply: base.ApplyFn,
                    params: hk.Params,
                    batch: base.Batch,
                    index: base.Index) -> Tuple[base.Array, base.LossMetrics]:
    loss, metrics = train_loss(apply, params, batch, index)
    for name, loss_fn in extra_losses.items():
      extra_loss, extra_metrics = loss_fn(apply, params, batch, index)
      metrics[f'{name}:loss'] = extra_loss
      for key, value in extra_metrics.items():
        metrics[f'{name}:{key}'] = value
    return loss, metrics

  return single_index.FunctionalSingleIndexLoss(combined_loss)


def combine_losses_as_metric(
    train_loss: base.LossFn,
    extra_losses: Dict[str, base.LossFn],
) -> base.LossFn:
  """Combines train_loss for training with extra_losses in metrics."""

  def combined_loss(enn: base.EpistemicNetwork,
                    params: hk.Params,
                    batch: base.Batch,
                    key: base.RngKey) -> Tuple[base.Array, base.LossMetrics]:
    loss, metrics = train_loss(enn, params, batch, key)
    for name, loss_fn in extra_losses.items():
      extra_loss, extra_metrics = loss_fn(enn, params, batch, key)
      metrics[f'{name}:loss'] = extra_loss
      for key, value in extra_metrics.items():
        metrics[f'{name}:{key}'] = value
    return loss, metrics

  return combined_loss
