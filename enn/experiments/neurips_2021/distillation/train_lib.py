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
"""Library for helper functions developing distillation ENN."""

import dataclasses
from typing import Dict, Sequence, Tuple

from enn import base_legacy as enn_base
from enn import losses
from enn import utils
import haiku as hk
import jax
import jax.numpy as jnp


@dataclasses.dataclass
class DistillationIndexer(enn_base.EpistemicIndexer):
  indexer: enn_base.EpistemicIndexer

  @property
  def mean_index(self) -> enn_base.Index:
    return -1

  def __call__(self, key: enn_base.RngKey) -> enn_base.Index:
    return self.indexer(key)


def _merge_params(params_seq: Sequence[hk.Params]) -> hk.Params:
  """Unsafe way to combine parameters in Haiku."""
  holding_dict = {}
  # TODO(author2): Look for safe/upstream version in Haiku.
  for params in params_seq:
    holding_dict.update(hk.data_structures.to_mutable_dict(params))
  return hk.data_structures.to_immutable_dict(holding_dict)


class DistillRegressionMLP(enn_base.EpistemicNetwork):
  """Add an extra MLP predicting (mean, log_var) to ENN extra output."""

  def __init__(self,
               enn: enn_base.EpistemicNetwork,
               hidden_sizes: Sequence[int] = (50, 50)):
    """Add an extra MLP predicting (mean, log_var) to ENN extra output."""

    def net_fn(x: enn_base.Array) -> Dict[str, enn_base.Array]:
      mean = hk.nets.MLP(list(hidden_sizes) + [1], name='distill_mean')
      var = hk.nets.MLP(list(hidden_sizes) + [1], name='distill_var')
      return {'mean': mean(x), 'log_var': var(x)}
    transformed = hk.without_apply_rng(hk.transform(net_fn))

    def apply(params: hk.Params,
              x: enn_base.Array,
              z: enn_base.Index) -> enn_base.OutputWithPrior:
      net_out = enn.apply(params, x, z)
      if not isinstance(net_out, enn_base.OutputWithPrior):
        net_out = enn_base.OutputWithPrior(net_out)
      net_out: enn_base.OutputWithPrior = net_out
      net_out.extra.update(transformed.apply(params, x))
      return net_out

    def init(key: enn_base.RngKey,
             x: enn_base.Array,
             z: enn_base.Index) -> hk.Params:
      base_params = enn.init(key, x, z)
      distill_params = transformed.init(key, x)
      return _merge_params([base_params, distill_params])

    indexer = DistillationIndexer(enn.indexer)

    super().__init__(apply, init, indexer)


@dataclasses.dataclass
class DistillRegressionLoss(enn_base.LossFn):
  """Distills mean and variance targets to extra components."""
  num_fake_batch: int
  num_index_sample: int
  only_real_data: bool = False

  def __call__(
      self,
      enn: enn_base.EpistemicNetwork,
      params: hk.Params,
      batch: enn_base.Batch,
      key: enn_base.RngKey,
  ) -> Tuple[enn_base.Array, enn_base.LossMetrics]:
    """Distills mean and variance targets to extra components."""
    if self.only_real_data:
      x = batch.x
    else:
      x = jax.random.normal(key, [self.num_fake_batch, batch.x.shape[1]])
    batched_out = losses.generate_batched_forward_at_data(
        self.num_index_sample, x, enn, params, key)
    batched_out: enn_base.OutputWithPrior = jax.lax.stop_gradient(batched_out)
    if hasattr(enn.indexer, 'mean_index'):
      distill_out = enn.apply(params, x, enn.indexer.mean_index)  # pytype:disable=attribute-error
      loss = kl_gauss(batched_out, distill_out)
      return jnp.mean(loss), {}
    else:
      raise ValueError(f'Indexer {enn.indexer} has no mean_index.')


def kl_gauss(batched_out: enn_base.OutputWithPrior,
             distill_out: enn_base.OutputWithPrior) -> enn_base.Array:
  batched_out = jax.lax.stop_gradient(batched_out)
  observed_mean = jnp.mean(utils.parse_net_output(batched_out), axis=0)
  observed_var = jnp.var(utils.parse_net_output(batched_out), axis=0)
  mean = distill_out.extra['mean']
  log_var = distill_out.extra['log_var']
  log_term = log_var - jnp.log(observed_var)
  mean_term = (observed_var + (mean - observed_mean) ** 2) / jnp.exp(log_var)
  return 0.5 * (log_term + mean_term - 1)


def combine_losses(loss_seq: Sequence[enn_base.LossFn]) -> enn_base.LossFn:
  """Combines a sequence of losses as a sum."""
  def combined_loss_fn(
      enn: enn_base.EpistemicNetwork,
      params: hk.Params,
      batch: enn_base.Batch,
      key: enn_base.RngKey,
  ) -> Tuple[enn_base.Array, enn_base.LossMetrics]:
    combined_loss = 0.
    combined_metrics = {}
    for loss_fn in loss_seq:
      loss, metrics = loss_fn(enn, params, batch, key)
      combined_loss += loss
      combined_metrics.update(metrics)
    return combined_loss, combined_metrics
  return combined_loss_fn
