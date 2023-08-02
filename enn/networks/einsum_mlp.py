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
"""Efficient ensemble implementations for JAX/Haiku via einsum."""
from typing import Callable, Optional, Sequence, Tuple

import chex
from enn import base
from enn.networks import base as networks_base
from enn.networks import indexers
from enn.networks import priors
from enn.networks import utils as network_utils
import haiku as hk
import jax
import jax.numpy as jnp

# TODO(author2): Delete this implementation, base ensemble is fast enough.


def make_einsum_ensemble_mlp_enn(
    output_sizes: Sequence[int],
    num_ensemble: int,
    nonzero_bias: bool = True,
    activation: Callable[[chex.Array], chex.Array] = jax.nn.relu,
) -> networks_base.EnnArray:
  """Factory method to create fast einsum MLP ensemble ENN.

  This is a specialized implementation for ReLU MLP without a prior network.

  Args:
    output_sizes: Sequence of integer sizes for the MLPs.
    num_ensemble: Integer number of elements in the ensemble.
    nonzero_bias: Whether to make the initial layer bias nonzero.
    activation: Jax callable defining activation per layer.
  Returns:
    EpistemicNetwork as an ensemble of MLP.
  """

  def ensemble_forward(x: chex.Array) -> networks_base.OutputWithPrior:
    """Forwards the entire ensemble at given input x."""
    model = EnsembleMLP(output_sizes, num_ensemble, nonzero_bias, activation)
    return model(x)  # pytype: disable=bad-return-type  # jax-ndarray

  transformed = hk.without_apply_rng(hk.transform(ensemble_forward))

  # Apply function selects the appropriate index of the ensemble output.
  def apply(params: hk.Params, x: chex.Array,
            z: base.Index) -> networks_base.OutputWithPrior:
    net_out = transformed.apply(params, x)
    one_hot_index = jax.nn.one_hot(z, num_ensemble)
    return jnp.dot(net_out, one_hot_index)

  def init(key: chex.PRNGKey, x: chex.Array,
           z: base.Index) -> hk.Params:
    del z
    return transformed.init(key, x)

  indexer = indexers.EnsembleIndexer(num_ensemble)

  # TODO(author3): Change apply and init fns above to work with state.
  apply = network_utils.wrap_apply_no_state_as_apply(apply)
  init = network_utils.wrap_init_no_state_as_init(init)
  return networks_base.EnnArray(apply, init, indexer)


def make_ensemble_mlp_with_prior_enn(
    output_sizes: Sequence[int],
    dummy_input: chex.Array,
    num_ensemble: int,
    prior_scale: float = 1.,
    nonzero_bias: bool = True,
    seed: int = 999,
) -> networks_base.EnnArray:
  """Factory method to create fast einsum MLP ensemble with matched prior.

  Args:
    output_sizes: Sequence of integer sizes for the MLPs.
    dummy_input: Example x input for prior initialization.
    num_ensemble: Integer number of elements in the ensemble.
    prior_scale: Float rescaling of the prior MLP.
    nonzero_bias: Whether to make the initial layer bias nonzero.
    seed: integer seed for prior init.

  Returns:
    EpistemicNetwork ENN of the ensemble of MLP with matches prior.
  """

  enn = make_einsum_ensemble_mlp_enn(output_sizes, num_ensemble, nonzero_bias)
  init_key, _ = jax.random.split(jax.random.PRNGKey(seed))
  prior_params, prior_state = enn.init(init_key, dummy_input, jnp.array([]))

  # Apply function selects the appropriate index of the ensemble output.
  def apply_with_prior(
      params: hk.Params,
      state: hk.State,
      x: chex.Array,
      z: base.Index,
  ) -> Tuple[networks_base.OutputWithPrior, hk.State]:
    ensemble_train, state = enn.apply(params, state, x, z)
    ensemble_prior, _ = enn.apply(prior_params, prior_state, x, z)
    output = networks_base.OutputWithPrior(
        train=ensemble_train, prior=ensemble_prior * prior_scale)
    return output, state

  return networks_base.EnnArray(apply_with_prior, enn.init, enn.indexer)


def make_ensemble_mlp_regularized_towards_prior(
    output_sizes: Sequence[int],
    dummy_input: chex.Array,
    num_ensemble: int,
    output_scale: float = 1.,
    nonzero_bias: bool = True,
    seed: int = 999,
) -> networks_base.EnnArray:
  """Factory method to create fast einsum MLP ensemble with matched prior.

  Args:
    output_sizes: Sequence of integer sizes for the MLPs.
    dummy_input: Example x input for prior initialization.
    num_ensemble: Integer number of elements in the ensemble.
    output_scale: Float rescaling of the output of the network.
    nonzero_bias: Whether to make the initial layer bias nonzero.
    seed: integer seed for prior init.

  Returns:
    EpistemicNetwork ENN of the ensemble of MLP with matches prior.
  """

  enn = make_einsum_ensemble_mlp_enn(output_sizes, num_ensemble, nonzero_bias)
  init_key, _ = jax.random.split(jax.random.PRNGKey(seed))
  prior_params, _ = enn.init(init_key, dummy_input, jnp.array([]))
  prior_params = jax.lax.stop_gradient(prior_params)
  # Apply function selects the appropriate index of the ensemble output.
  def apply_with_prior(
      params: hk.Params,
      state: hk.State,
      x: chex.Array,
      z: base.Index,
  ) -> Tuple[chex.Array, hk.State]:
    combined_params = jax.tree_map(lambda p1, p2: p1+p2, params, prior_params)
    ensemble_train, state = enn.apply(combined_params, state, x, z)
    output = ensemble_train * output_scale
    return output, state

  return networks_base.EnnArray(apply_with_prior, enn.init, enn.indexer)


# TODO(author3): Come up with a better name and use ensembles.py instead.
def make_ensemble_prior(output_sizes: Sequence[int],
                        num_ensemble: int,
                        dummy_input: chex.Array,
                        seed: int = 999,) -> priors.PriorFn:
  """Combining a few ensemble elements as prior function."""
  def net_fn(x):
    model = EnsembleMLP(output_sizes, num_ensemble)
    return model(x)
  transformed = hk.without_apply_rng(hk.transform(net_fn))
  rng = hk.PRNGSequence(seed)
  params = transformed.init(next(rng), dummy_input)
  prior_fn = lambda x, z: jnp.dot(transformed.apply(params, x), z)
  return jax.jit(prior_fn)
################################################################################
# Einsum implementation of MLP


class EnsembleBranch(hk.Module):
  """Branches a single linear layer to num_ensemble, output_size."""

  def __init__(self,
               num_ensemble: int,
               output_size: int,
               nonzero_bias: bool,
               w_init: Optional[hk.initializers.Initializer] = None,
               name: str = 'ensemble_branch'):
    super().__init__(name=name)
    self.num_ensemble = num_ensemble
    self.output_size = output_size
    self.nonzero_bias = nonzero_bias
    self.w_init = w_init

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:  # [B, H] -> [B, D, K]
    assert inputs.ndim == 2
    unused_batch, input_size = inputs.shape

    if self.nonzero_bias:
      b_init = hk.initializers.TruncatedNormal(
          stddev=(1. / jnp.sqrt(input_size)))
    else:
      b_init = jnp.zeros

    if self.w_init is not None:
      w_init = self.w_init
    else:
      w_init = hk.initializers.TruncatedNormal(
          stddev=(1. / jnp.sqrt(input_size)))

    w = hk.get_parameter(
        'w', [input_size, self.output_size, self.num_ensemble], init=w_init)
    b = hk.get_parameter(
        'b', [self.output_size, self.num_ensemble], init=b_init)

    return jnp.einsum('bi,ijk->bjk', inputs, w) + jnp.expand_dims(b, axis=0)


class EnsembleLinear(hk.Module):
  """Keeps num_ensemble linear layers in parallel without interactions."""

  def __init__(self,
               output_size: int,
               w_init: Optional[hk.initializers.Initializer] = None,
               name: str = 'linear'):
    super().__init__(name=name)
    self.output_size = output_size
    self.w_init = w_init

  def __call__(self,
               inputs: jnp.ndarray) -> jnp.ndarray:  # [B, H, K] -> [B. D, K]
    assert inputs.ndim == 3
    unused_batch, input_size, self.num_ensemble = inputs.shape

    if self.w_init is not None:
      w_init = self.w_init
    else:
      w_init = hk.initializers.TruncatedNormal(
          stddev=(1. / jnp.sqrt(input_size)))

    w = hk.get_parameter(
        'w', [input_size, self.output_size, self.num_ensemble], init=w_init)
    b = hk.get_parameter(
        'b', [self.output_size, self.num_ensemble], init=jnp.zeros)
    return jnp.einsum('bik,ijk->bjk', inputs, w) + jnp.expand_dims(b, axis=0)


class EnsembleMLP(hk.Module):
  """Parallel num_ensemble MLPs all with same output_sizes.

  In the first layer, the input is 'branched' to num_ensemble linear layers.
  Then, in subsequent layers it is purely parallel EnsembleLinear.
  """

  def __init__(self,
               output_sizes: Sequence[int],
               num_ensemble: int,
               nonzero_bias: bool = True,
               activation: Callable[[chex.Array], chex.Array] = jax.nn.relu,
               w_init: Optional[hk.initializers.Initializer] = None,
               name: str = 'ensemble_mlp'):
    super().__init__(name=name)
    self.num_ensemble = num_ensemble
    self.activation = activation

    layers = []
    for index, output_size in enumerate(output_sizes):
      if index == 0:
        layers.append(
            EnsembleBranch(num_ensemble, output_size, nonzero_bias, w_init))
      else:
        layers.append(EnsembleLinear(output_size, w_init))
    self.layers = tuple(layers)

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:  # [B, H] -> [B, D, K]
    num_layers = len(self.layers)
    out = hk.Flatten()(inputs)
    for i, layer in enumerate(self.layers):
      out = layer(out)
      if i < num_layers - 1:
        out = self.activation(out)
    return out  # pytype: disable=bad-return-type  # numpy-scalars
