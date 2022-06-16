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
"""Network definitions for epinet.

Trying to fork out some reusable pieces for the code.
"""

from typing import Optional, Sequence

from enn import base_legacy as enn_base
from enn.networks import indexers
from enn.networks import mlp
import haiku as hk


def make_mlp_epinet(
    output_sizes: Sequence[int],
    epinet_hiddens: Sequence[int],
    index_dim: int,
    expose_layers: Optional[Sequence[bool]] = None,
    prior_scale: float = 1.) -> enn_base.EpistemicNetworkWithState:
  """Factory method to create a standard MLP epinet."""

  def net_fn(x: enn_base.Array, z: enn_base.Index) -> enn_base.OutputWithPrior:
    base_mlp = mlp.ExposedMLP(output_sizes, expose_layers, name='base_mlp')
    num_classes = output_sizes[-1]
    train_epinet = mlp.ProjectedMLP(
        epinet_hiddens, num_classes, index_dim, name='train_epinet')
    prior_epinet = mlp.ProjectedMLP(
        epinet_hiddens, num_classes, index_dim, name='prior_epinet')

    base_out = base_mlp(x)
    features = base_out.extra['exposed_features']
    epi_train = train_epinet(features, z)
    epi_prior = prior_epinet(features, z)
    return enn_base.OutputWithPrior(
        train=base_out.train + epi_train,
        prior=prior_scale * epi_prior,
    )

  transformed = hk.without_apply_rng(hk.transform_with_state(net_fn))
  return enn_base.EpistemicNetworkWithState(
      apply=transformed.apply,
      init=transformed.init,
      indexer=indexers.GaussianIndexer(index_dim),
  )
