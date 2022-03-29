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
"""Network definitions for epinet.

Trying to fork out some reusable pieces for the code.
"""

from typing import Optional, Sequence

import chex
from enn import base as enn_base
from enn.networks import einsum_mlp
from enn.networks import indexers
import haiku as hk
import jax
import jax.numpy as jnp


class ExposedMLP(enn_base.EpistemicModule):
  """MLP module that exposes internal layers in output."""

  def __init__(self,
               output_sizes: Sequence[int],
               stop_gradient: bool = True,
               name: Optional[str] = None):
    """ReLU MLP that also exposes the internals as output."""
    super().__init__(name=name)
    layers = []
    for index, output_size in enumerate(output_sizes):
      layers.append(hk.Linear(output_size, name=f'linear_{index}'))
    self.layers = tuple(layers)
    self.num_layers = len(self.layers)
    self.output_size = output_sizes[-1]
    self.stop_gradient = stop_gradient

  def __call__(self, inputs: enn_base.Array) -> enn_base.OutputWithPrior:
    """Standard MLP but exposes 'all_features' in .extra output."""
    all_features = [inputs]
    out = inputs
    for i, layer in enumerate(self.layers):
      out = layer(out)
      if i < (self.num_layers - 1):
        out = jax.nn.relu(out)
      all_features.append(out)

    all_features = jnp.concatenate(all_features, axis=1)
    if self.stop_gradient:
      all_features = jax.lax.stop_gradient(all_features)
    extra = {'all_features': all_features}
    return enn_base.OutputWithPrior(train=out, extra=extra)


class ProjectedMLP(enn_base.EpistemicModule):
  """MLP whose output in the final layer is then dot-product with Z-index."""

  def __init__(self,
               hidden_sizes: Sequence[int],
               final_out: int,
               index_dim: int,
               concat_index: bool = True,
               name: Optional[str] = None):
    super().__init__(name=name)
    self.hidden_sizes = hidden_sizes
    self.final_out = final_out
    self.index_dim = index_dim
    self.concat_index = concat_index
    output_sizes = list(self.hidden_sizes) + [self.final_out * index_dim]
    self.mlp = hk.nets.MLP(output_sizes)

  def __call__(self,
               inputs: enn_base.Array,
               index: enn_base.Index) -> enn_base.Array:
    chex.assert_shape(index, [self.index_dim])

    if self.concat_index:
      # Concatenate the index z to the *inputs* as well.
      batched_z = jnp.repeat(jnp.expand_dims(index, 0), inputs.shape[0], axis=0)
      inputs = jnp.concatenate([batched_z, inputs], axis=1)
    reshaped_output = jnp.reshape(
        self.mlp(inputs), [inputs.shape[0], self.final_out, self.index_dim])
    return jnp.dot(reshaped_output, index)


def make_mlp_epinet(output_sizes: Sequence[int],
                    epinet_hiddens: Sequence[int],
                    index_dim: int,
                    prior_scale: float = 1.) -> enn_base.EpistemicNetwork:
  """Factory method to create a standard MLP epinet."""

  def net_fn(x: enn_base.Array, z: enn_base.Index) -> enn_base.OutputWithPrior:
    base_mlp = ExposedMLP(output_sizes, name='base_mlp')
    train_epinet = ProjectedMLP(
        epinet_hiddens, output_sizes[-1], index_dim, name='train_epinet')
    prior_epinet = ProjectedMLP(
        epinet_hiddens, output_sizes[-1], index_dim, name='prior_epinet')

    base_out = base_mlp(x)
    epi_train = train_epinet(base_out.extra['all_features'], z)
    epi_prior = prior_epinet(base_out.extra['all_features'], z)
    return enn_base.OutputWithPrior(
        train=base_out.train + epi_train,
        prior=prior_scale * epi_prior,
    )

  transformed = hk.without_apply_rng(hk.transform(net_fn))
  return enn_base.EpistemicNetwork(
      apply=transformed.apply,
      init=transformed.init,
      indexer=indexers.GaussianIndexer(index_dim)
  )


def make_mlp_epinet_with_ensemble_prior(
    output_sizes: Sequence[int],
    epinet_hiddens: Sequence[int],
    ensemble_hiddens: Sequence[int],
    index_dim: int,
    prior_scale: float = 1.) -> enn_base.EpistemicNetwork:
  """Factory method to create a standard MLP epinet."""

  def net_fn(x: enn_base.Array, z: enn_base.Index) -> enn_base.OutputWithPrior:
    # TODO(author3): Use ensembles.py instead of einsum_mlp.py
    ensemble_prior = einsum_mlp.EnsembleMLP(
        list(ensemble_hiddens) + [output_sizes[-1]],
        index_dim, name='ensemble_prior')
    ensemble_prior_out = jnp.dot(ensemble_prior(x), z)

    base_mlp = ExposedMLP(output_sizes, name='base_mlp')
    base_out = base_mlp(x)

    train_epinet = ProjectedMLP(
        epinet_hiddens, output_sizes[-1], index_dim, name='train_epinet')
    epi_train = train_epinet(base_out.extra['all_features'], z)

    return enn_base.OutputWithPrior(
        train=base_out.train + epi_train,
        prior=prior_scale * ensemble_prior_out,
    )

  transformed = hk.without_apply_rng(hk.transform(net_fn))
  return enn_base.EpistemicNetwork(
      apply=transformed.apply,
      init=transformed.init,
      indexer=indexers.GaussianIndexer(index_dim)
  )
