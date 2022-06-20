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
import haiku as hk
import typing_extensions


# TODO(author3): Clean-up the names.
# TODO(author3): Mention about the wrappers.
class SingleLossFn(
    typing_extensions.Protocol[base.Input, base.Data]):
  """Calculates a loss based on one batch of data per index.

  You can use utils.average_single_index_loss to make a LossFnArray out of
  the SingleLossFnArray.
  """

  def __call__(
      self,
      apply: base.ApplyFn[base.Input],
      params: hk.Params,
      state: hk.State,
      batch: base.Data,
      index: base.Index,
  ) -> base.LossOutput:
    """Computes a loss based on one batch of data and one index."""


# base.LossFn specialized to work only with Array inputs and Batch data.
LossFnArray = base.LossFn[chex.Array, base.Batch]
SingleLossFnArray = SingleLossFn[chex.Array, base.Batch]

# Supporting loss functions without state
LossOutputNoState = Tuple[chex.Array, base.LossMetrics]


class LossFnNoState(typing_extensions.Protocol):
  """Calculates a loss based on one batch of data per random key."""

  def __call__(self,
               enn: networks.EnnNoState,
               params: hk.Params,
               batch: base.Batch,
               key: chex.PRNGKey) -> LossOutputNoState:
    """Computes a loss based on one batch of data and a random key."""


class SingleLossFnNoState(typing_extensions.Protocol):
  """Calculates a loss based on one batch of data per index.

  You can use utils.average_single_index_loss to make a LossFn out of the
  SingleLossFn.
  """

  def __call__(self,
               apply: networks.ApplyNoState,
               params: hk.Params,
               batch: base.Batch,
               index: base.Index) -> LossOutputNoState:
    """Computes a loss based on one batch of data and one index."""

