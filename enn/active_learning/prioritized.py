# pylint: disable=g-bad-file-header
# Copyright 2023 DeepMind Technologies Limited. All Rights Reserved.
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
"""An active learner that uses priority functions to select data."""

import typing as tp

import chex
from enn import datasets
from enn import networks
from enn.active_learning import base
from enn.active_learning import priorities
import haiku as hk
import jax
import jax.numpy as jnp


class PrioritizedBatcher(base.ActiveLearner):
  """Prioritizes bathces based on a priority fn."""

  def __init__(
      self,
      enn_batch_fwd: networks.EnnBatchFwd[chex.Array],
      acquisition_size: int = 64,
      priority_fn_ctor: tp.Optional[base.PriorityFnCtor] = None,
  ):
    """Initializes the batcher."""
    self._acquisition_size = acquisition_size

    if priority_fn_ctor is None:
      priority_fn_ctor = priorities.get_priority_fn_ctor('uniform')
    self._priority_fn = priority_fn_ctor(enn_batch_fwd)

  def sample_batch(
      self,
      params: hk.Params,
      state: hk.State,
      batch: datasets.ArrayBatch,
      key: chex.PRNGKey,
  ) -> datasets.ArrayBatch:
    """Acquires data. This is the function per device (can get pmaped)."""
    pool_size = len(batch.y)

    candidate_scores, unused_metrics = self._priority_fn(
        params, state, batch, key)
    # Cannot acquire data more than batch size
    acquisition_size = min(self._acquisition_size, pool_size)

    selected_idxs = jnp.argsort(candidate_scores)[-acquisition_size:]
    acquired_data = get_at_index(batch, selected_idxs)

    return acquired_data

  @property
  def acquisition_size(self):
    """Return the acquisition size for the active learner."""
    return self._acquisition_size

  @acquisition_size.setter
  def acquisition_size(self, size: int) -> None:
    """Overwrites the acquisition size. Useful when we pmap sample_batch."""
    self._acquisition_size = size


_T = tp.TypeVar('_T')


def get_at_index(t: _T, idx: chex.Array) -> _T:
  """Gets values at the indices specified by idx array."""
  return jax.tree_map(lambda x: x[idx], t)
