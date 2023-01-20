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
"""Implementing Dropout as an ENN in JAX."""
from typing import List, Optional, Sequence

import chex
from enn import base
from enn.networks import base as networks_base
from enn.networks import indexers
from enn.networks import utils as network_utils
import haiku as hk
import jax
import jax.numpy as jnp


def cummulative_sum(x):
  """Returns the cummulative sum of elements of a list."""
  # Create a copy of list x.
  output = x[:]
  for i in range(len(output) - 1):
    output[i + 1] += output[i]
  return output


def generate_masks(key: chex.PRNGKey, input_size: int,
                   output_sizes: Sequence[int],
                   dropout_rate: float) -> List[jnp.ndarray]:
  """Generates masks for nodes of the network excluding nodes of the last layer."""
  # Create a list of num nodes per layer, ignoring the output layer.
  num_nodes_layers = [input_size,] + list(output_sizes[:-1])
  keep_rate = 1.0 - dropout_rate
  # Create an array of masks, one mask for each nodes of the network, excluding
  # nodes of the last layer.
  masks = jax.random.bernoulli(
      key, keep_rate, shape=[sum(num_nodes_layers)])
  masks /= keep_rate
  # Create a list where each element is an array of masks for one layer.
  # TODO(author3): Use jnp.cumsum instead of cummulative_sum
  masks = jnp.split(masks, cummulative_sum(num_nodes_layers))

  return masks


class MLPDropoutENN(networks_base.EnnArray):
  """MLP with dropout as an ENN."""

  def __init__(
      self,
      output_sizes: Sequence[int],
      dropout_rate: float = 0.05,
      dropout_input: bool = True,
      seed: int = 0,
      nonzero_bias: bool = True,
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None
  ):
    """MLP with dropout as an ENN."""

    def enn_fn(inputs: chex.Array,
               z: base.Index) -> networks_base.Output:

      assert inputs.ndim == 2
      unused_batch, input_size = inputs.shape

      x = hk.Flatten()(inputs)

      # Generate a list of masks, one for each network layer, exclduing the
      # output layer
      masks = generate_masks(
          key=z,
          input_size=input_size,
          output_sizes=output_sizes,
          dropout_rate=dropout_rate)

      # Note that we consider a dropout layer after the input to be
      # consistent with the paper "Dropout as a Bayesian Approximation:
      # Representing Model Uncertainty in Deep Learning" (2015),
      # https://github.com/yaringal/DropoutUncertaintyExps/blob/master/net/net.py
      if dropout_input:
        mask = masks[0]
        x = jnp.einsum('bi,i->bi', x, mask)

      for layer_index, output_size in enumerate(output_sizes):
        if layer_index == 0:
          if nonzero_bias:
            b_init_0 = hk.initializers.TruncatedNormal(
                stddev=(1. / jnp.sqrt(input_size)))
          else:
            b_init_0 = b_init
          x = hk.Linear(output_size, w_init=w_init, b_init=b_init_0)(x)
        else:
          x = hk.Linear(output_size, w_init=w_init, b_init=b_init)(x)
        if layer_index < len(output_sizes) - 1:
          mask = masks[layer_index + 1]
          x = jnp.einsum('bi,i->bi', x, mask)
          x = jax.nn.relu(x)
      return x

    # Note that our enn_fn is stochastic because we generate masks in it. But,
    # since we pass a chex.PRNGKey directly to it, we can still wrap transformed
    # function with hk.without_apply_rng.
    transformed = hk.without_apply_rng(hk.transform(enn_fn))

    # We use a simple indexer which is basically an identity map.
    indexer = indexers.PrngIndexer()

    # Apply function for enn_fn requires a rng key to generate masks. We use
    # the index z in f(x,z) as the rng key.
    def apply(params: hk.Params, x: chex.Array,
              z: base.Index) -> networks_base.Output:
      net_out = transformed.apply(params, x, z)
      return net_out

    apply = network_utils.wrap_apply_no_state_as_apply(apply)
    init = network_utils.wrap_init_no_state_as_init(transformed.init)

    super().__init__(apply, init, indexer)
