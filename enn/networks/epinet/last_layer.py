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
"""Helper to get a model given collection of models."""

from typing import Optional, Sequence

import chex
from enn import base_legacy as enn_base
from enn import networks
from enn.networks.epinet import base as epinet_base
import haiku as hk
import jax.numpy as jnp


class MLPEpinetWithPrior(epinet_base.EpinetWithState):
  """MLP epinet with matching prior function."""

  def __init__(self,
               index_dim: int,
               num_classes: int,
               epinet_hiddens: Sequence[int],
               prior_epinet_hiddens: Optional[Sequence[int]] = None,
               prior_scale: float = 1,
               drop_inputs: bool = False):
    """Defines an MLP epinet with matching prior function."""
    if prior_epinet_hiddens is None:
      prior_epinet_hiddens = epinet_hiddens

    def epinet_fn(inputs: enn_base.Array,
                  index: enn_base.Index,
                  hidden: enn_base.Array) -> enn_base.OutputWithPrior:
      # Creating networks
      train_epinet = networks.ProjectedMLP(
          epinet_hiddens, num_classes, index_dim, name='train_epinet')
      prior_epinet = networks.ProjectedMLP(
          prior_epinet_hiddens, num_classes, index_dim, name='prior_epinet')

      if drop_inputs:
        epi_inputs = hidden
      else:
        flat_inputs = hk.Flatten()(inputs)
        epi_inputs = jnp.concatenate([hidden, flat_inputs], axis=1)

      # Wiring networks: add linear epinet (+ prior) from final output layer.
      epi_train = train_epinet(epi_inputs, index)
      epi_prior = prior_epinet(epi_inputs, index)
      return enn_base.OutputWithPrior(
          train=epi_train,
          prior=prior_scale * epi_prior,
      )

    # Form ENN from haiku transformed.
    transformed = hk.without_apply_rng(hk.transform_with_state(epinet_fn))
    indexer = networks.GaussianIndexer(index_dim)
    super().__init__(transformed.apply, transformed.init, indexer)


def parse_base_hidden(
    base_out: enn_base.Output,
    hidden_name: str = 'final_out',
) -> chex.Array:
  """Parses the final hidden layer from the base network output."""
  # TODO(author2): improve type checking on base_out
  assert isinstance(base_out, enn_base.OutputWithPrior)
  assert hidden_name in base_out.extra
  return base_out.extra[hidden_name]
