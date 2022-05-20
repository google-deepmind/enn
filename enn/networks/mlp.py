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
"""MLP variants for ENN."""

from typing import Optional, Sequence

import chex
from enn import base as enn_base
import haiku as hk
import jax
import jax.numpy as jnp


class ExposedMLP(hk.Module):
  """MLP module that exposes internal layers in output."""

  def __init__(self,
               output_sizes: Sequence[int],
               expose_layers: Optional[Sequence[bool]] = None,
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
    self.expose_layers = expose_layers
    # if expose_layers is None, we expose all layers
    if self.expose_layers is None:
      self.expose_layers = [True] * len(output_sizes)
    assert len(self.expose_layers) == len(self.layers)

  def __call__(self, inputs: enn_base.Array) -> enn_base.OutputWithPrior:
    """Standard MLP but exposes 'exposed_features' in .extra output."""
    layers_features = []
    out = inputs
    for i, layer in enumerate(self.layers):
      out = layer(out)
      if i < (self.num_layers - 1):
        out = jax.nn.relu(out)
      layers_features.append(out)

    exposed_features = [inputs]
    for i, layer_feature in enumerate(layers_features):
      # Add this layer feature if the expose flag for this layer is True
      if self.expose_layers[i]:
        exposed_features.append(layer_feature)

    exposed_features = jnp.concatenate(exposed_features, axis=1)
    if self.stop_gradient:
      exposed_features = jax.lax.stop_gradient(exposed_features)
    extra = {'exposed_features': exposed_features}
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

