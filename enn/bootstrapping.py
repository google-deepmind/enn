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

"""Utilities for generating bootstrap weights in JAX.

Note that we *may* want to pull this into a separate library from the ENN.
"""

from typing import Callable, Optional, Sequence

from absl import logging
import chex
from enn import base
from enn import networks
import jax
import jax.numpy as jnp
import typing_extensions

BatchWeights = base.Array  # Bootstrap weights for each datapoint
BootstrapFn = Callable[[base.DataIndex, base.Index], BatchWeights]

# TODO(author2): Currently all functions written assuming batch dimensions.
# but it might be more elegant to rework the vmap and instead define for one
# example at a time.
# batch_weights = boot_fn(data_index, index)  # (batch_size, 1) shape

# TODO(author2): Refactor batch_weights to be just (batch_size,) shape.


class WeightFn(typing_extensions.Protocol):
  """Interface for weight-generating functions."""

  def __call__(
      self,
      rng_key: base.RngKey,
      indices: Optional[Sequence[int]] = None,
  ) -> jnp.DeviceArray:
    ...


DISTRIBUTIONS = {
    'poisson': lambda x, shape=(): jax.random.poisson(x, 1, shape=shape),
    'exponential':
        lambda x, shape=(): jax.random.exponential(x, shape=shape),
    'bernoulli':
        lambda x, shape=(): 2 * jax.random.bernoulli(x, 0.5, shape=shape),
    'uniform': lambda x, shape=(): 2 * jax.random.uniform(x, shape=shape),
}


def null_bootstrap(
    data_index: base.DataIndex, index: base.Index) -> BatchWeights:
  """Null bootstrap does not reweight the data at all."""
  del index
  chex.assert_shape(data_index, (None, 1))
  return jnp.ones_like(data_index)


def make_boot_fn(enn: base.EpistemicNetwork,
                 distribution: str,
                 seed: int = 0) -> BootstrapFn:
  """Factory method to create bootstrap for ensemble = integer indices."""
  # None works as a special case for no function
  if distribution == 'none' or distribution is None:
    return null_bootstrap

  # Bootstrapping for ensemble/discrete options
  if isinstance(enn.indexer, networks.EnsembleIndexer):
    if distribution not in DISTRIBUTIONS:
      raise ValueError(f'dist={distribution} not implemented for ensemble.')
    weight_fn = DISTRIBUTIONS[distribution]
    return _make_ensemble_bootstrap_fn(weight_fn, seed)

  # Bootstrapping for Gaussian with unit index
  elif isinstance(enn.indexer, networks.GaussianWithUnitIndexer):
    index_dim = enn.indexer.index_dim
    logging.warning(
        'WARNING: indexer is in development, bootstrap may not be correct.')
    if distribution == 'bernoulli':
      return _make_gaussian_index_bernoulli_bootstrap(index_dim, seed)
    else:
      raise ValueError(
          f'dist={distribution} not implemented for GaussianIndexer.')

  # Bootstrapping for Scaled Gaussian index
  elif isinstance(enn.indexer, networks.ScaledGaussianIndexer):
    index_dim = enn.indexer.index_dim
    if distribution == 'bernoulli':
      return _make_gaussian_index_bernoulli_bootstrap(index_dim, seed)
    elif distribution == 'exponential':
      return _make_gaussian_index_exponential_bootstrap(index_dim, seed, 1)
    else:
      raise ValueError(
          f'dist={distribution} not implemented for GaussianIndexer.')

  # Bootstrapping for PRNG index
  elif isinstance(enn.indexer, networks.PrngIndexer):
    if distribution not in DISTRIBUTIONS:
      raise ValueError(f'dist={distribution} not implemented for gauss_enn.')
    weight_fn = DISTRIBUTIONS[distribution]
    return _make_prng_bootstrap_fn(weight_fn)

  else:
    raise ValueError(
        f'Bootstrapping for EpistemicIndexer={enn.indexer} not implemented.')


def _make_prng_bootstrap_fn(weight_fn: WeightFn) -> BootstrapFn:
  """Factory method for bootstrap with PRNG index."""
  def boot_fn(data_index: base.DataIndex, index: base.Index):
    chex.assert_shape(data_index, (None, 1))
    boot_weights = weight_fn(index, data_index.shape)
    return boot_weights[:, None]
  return boot_fn


def _make_key(data_index: base.Array, seed: int) -> base.RngKey:
  """Creates RngKeys for a batch of data index."""
  chex.assert_shape(data_index, (None, 1))
  return jax.vmap(jax.random.PRNGKey)(jnp.squeeze(data_index, axis=1) + seed)


def _make_ensemble_bootstrap_fn(
    weight_fn: WeightFn, seed: int = 0) -> BootstrapFn:
  """Factory method to create bootstrapping function with ensemble index.

  Args:
    weight_fn: weight distribution function e.g. jax.random.exponential.
    seed: Optional integer added to the data_keys

  Returns:
    BootstrapFn appropriate for ensemble = assumes integer index.
  """
  fold_in = jax.vmap(jax.random.fold_in)
  weight_fn = jax.vmap(weight_fn)

  def boot_fn(data_index: base.DataIndex, index: base.Index):
    """Assumes integer index for ensemble weights."""
    chex.assert_shape(data_index, (None, 1))
    if not index.shape:  # If it's a single integer -> repeat for batch
      index = jnp.repeat(index, len(data_index))
    data_keys = _make_key(data_index, seed)
    rng_keys = fold_in(data_keys, index)
    return weight_fn(rng_keys)[:, None]

  return boot_fn


def _make_gaussian_index_exponential_bootstrap(
    index_dim: int,
    seed: int = 0,
    scale: float = 1,
    fold_seed: int = 666) -> BootstrapFn:
  """Factory method to create the approximate exponential weighting."""
  fold_in = jax.vmap(jax.random.fold_in, in_axes=[0, None])
  std_gauss = lambda x: jax.random.normal(x, [index_dim]) * scale
  sample_std_gaussian = jax.vmap(std_gauss)

  def boot_fn(data_index: base.DataIndex, index: base.Index):
    """Assumes integer index for ensemble weights."""
    chex.assert_shape(data_index, (None, 1))
    b_keys = _make_key(data_index, seed)
    b = sample_std_gaussian(b_keys)
    c_keys = fold_in(b_keys, fold_seed)
    c = sample_std_gaussian(c_keys)

    batch_size = data_index.shape[0]
    z = jnp.repeat(jnp.expand_dims(index, 0), batch_size, axis=0)
    weights = 0.5 * (jnp.sum(b * z, axis=1) ** 2 + jnp.sum(c * z, axis=1) ** 2)
    return weights[:, None]

  return boot_fn


def _make_gaussian_index_bernoulli_bootstrap(
    index_dim: int,
    seed: int = 0) -> BootstrapFn:
  """Factory method to create the approximate bernoulli weighting."""
  std_gauss = lambda x: jax.random.normal(x, [index_dim]) / jnp.sqrt(index_dim)
  sample_std_gaussian = jax.vmap(std_gauss)

  def boot_fn(data_index: base.DataIndex, index: base.Index):
    """Assumes integer index for ensemble weights."""
    chex.assert_shape(data_index, (None, 1))
    b_keys = _make_key(data_index, seed)
    b = sample_std_gaussian(b_keys)
    batch_size = data_index.shape[0]
    z = jnp.repeat(jnp.expand_dims(index, 0), batch_size, axis=0)
    weights = 1. + jnp.sign(jnp.sum(b * z, axis=1))
    return weights[:, None]

  return boot_fn
