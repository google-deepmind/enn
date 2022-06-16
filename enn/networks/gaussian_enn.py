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

"""Implementing a Gaussian ENN in JAX.

This model came out of brainstorming with who/benvanroy, who/iosband.
Each linear unit in the neural net is augmented:
  Wx + b --> Wx + b + c Z, for Z ~ N(0, 1)

You are adding a learnable bias for each segment of
This is an implementation framing that network as an ENN.
"""

from typing import Callable, Sequence

from enn import base_legacy
from enn import utils
from enn.networks import indexers
import haiku as hk
import haiku.experimental as hke
import jax
import jax.numpy as jnp


def _is_linear_bias(context: hke.ParamContext):
  return (context.full_name.endswith('/b')
          and isinstance(context.module, hk.Linear))


def make_enn_creator(init_scale: float = 1.):
  """Make enn_creator initializing c unit to init_scale."""
  custom_init = lambda shape, dtype: init_scale * jnp.ones(shape, dtype)
  def enn_creator(next_creator, shape, dtype, init, context):
    """Create gaussian enn linear layer."""
    # TODO(author2): How to import hk._src.base types correctly?
    if _is_linear_bias(context):  # Append gaussian bias term
      standard_bias = next_creator(shape, dtype, init)
      gaussian_bias = next_creator(shape, dtype, custom_init)
      return standard_bias, gaussian_bias
    else:  # Return the usual creator
      return next_creator(shape, dtype, init)
  return enn_creator


def enn_getter(next_getter, value, context):
  """Get variables for gaussian enn linear layer."""
  # TODO(author2): How to import hk._src.base types correctly?
  if _is_linear_bias(context):
    standard_bias = next_getter(value[0])
    gaussian_bias = next_getter(value[1])
    noise = jax.random.normal(hk.next_rng_key(), standard_bias.shape)
    return standard_bias + gaussian_bias * noise
  else:
    return next_getter(value)


class GaussianNoiseEnn(base_legacy.EpistemicNetworkWithState):
  """GaussianNoiseEnn from callable module."""

  def __init__(self,
               module_ctor: Callable[[], hk.Module],
               init_scale: float = 1.):
    """GaussianNoiseEnn from callable module."""
    enn_creator = make_enn_creator(init_scale=init_scale)

    def net_fn(inputs: base_legacy.Array) -> base_legacy.Array:
      with hke.custom_getter(enn_getter), hke.custom_creator(enn_creator):
        output = module_ctor()(inputs)  # pytype: disable=not-callable
        return output

    # TODO(author2): Note that the GaussianENN requires a rng_key in place of an
    # index. Therefore we do *not* hk.without_apply_rng.
    transformed = hk.transform(net_fn)
    apply = lambda params, x, z: transformed.apply(params, z, x)
    init = lambda rng, x, z: transformed.init(rng, x)

    # TODO(author3): Change apply and init fns above to work with state.
    apply = utils.wrap_apply_as_apply_with_state(apply)
    init = utils.wrap_init_as_init_with_state(init)

    super().__init__(apply, init, indexer=indexers.PrngIndexer(),)


class GaussianNoiseMLP(base_legacy.EpistemicNetworkWithState):
  """Gaussian Enn on a standard MLP."""

  def __init__(self, output_sizes: Sequence[int], init_scale: float = 1.):
    """Gaussian Enn on a standard MLP."""
    enn = GaussianNoiseEnn(lambda: hk.nets.MLP(output_sizes), init_scale)
    super().__init__(enn.apply, enn.init, enn.indexer)
