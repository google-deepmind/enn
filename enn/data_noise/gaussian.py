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

"""Utilities for perturbing data with Gaussian noise."""

import dataclasses
from typing import Callable, Union

import chex
from enn import base
from enn import networks
from enn.data_noise import base as data_noise_base
import jax
import jax.numpy as jnp

_ENN = Union[networks.EnnNoState,
             networks.EnnArray]


@dataclasses.dataclass
class GaussianTargetNoise(data_noise_base.DataNoise):
  """Apply Gaussian noise to the target y."""
  enn: _ENN
  noise_std: float
  seed: int = 0

  def __call__(self, data: base.Batch,
               index: base.Index) -> base.Batch:
    """Apply Gaussian noise to the target y."""
    chex.assert_shape(data.y, (None, 1))  # Only implemented for 1D now.
    noise_fn = make_noise_fn(self.enn, self.noise_std, self.seed)
    y_noise = noise_fn(data.data_index, index)
    return data._replace(y=data.y + y_noise)


NoiseFn = Callable[[base.DataIndex, base.Index],
                   chex.Array]


def make_noise_fn(enn: _ENN,
                  noise_std: float,
                  seed: int = 0) -> NoiseFn:
  """Factory method to create noise_fn for given ENN."""
  indexer = data_noise_base.get_indexer(enn.indexer)

  if isinstance(indexer, networks.EnsembleIndexer):
    return _make_ensemble_gaussian_noise(noise_std, seed)

  elif isinstance(indexer, networks.GaussianIndexer):
    return _make_gaussian_index_noise(indexer.index_dim, noise_std, seed)

  elif isinstance(indexer, networks.ScaledGaussianIndexer):
    return _make_scaled_gaussian_index_noise(indexer.index_dim, noise_std, seed)

  elif isinstance(indexer, networks.GaussianWithUnitIndexer):
    # Ignore the first component which is always 1 and not Gaussian.
    effective_index_dim = indexer.index_dim - 1
    raw_noise = _make_scaled_gaussian_index_noise(
        effective_index_dim, noise_std, seed)
    noise_fn = lambda d, z: raw_noise(d, z[1:])  # Don't include unit component.
    return noise_fn

  else:
    raise ValueError(f'Unsupported ENN={enn}.')


def _make_key(data_index: chex.Array, seed: int) -> chex.PRNGKey:
  """Creates RngKeys for a batch of data index."""
  chex.assert_shape(data_index, (None, 1))
  return jax.vmap(jax.random.PRNGKey)(jnp.squeeze(data_index, axis=1) + seed)


def _make_ensemble_gaussian_noise(noise_std: float, seed: int) -> NoiseFn:
  """Factory method to add Gaussian noise for ensemble index."""
  batch_fold_in = jax.vmap(jax.random.fold_in)
  batch_normal = jax.vmap(jax.random.normal)

  def noise_fn(data_index: base.DataIndex,
               index: base.Index) -> chex.Array:
    """Assumes integer index for ensemble."""
    chex.assert_shape(data_index, (None, 1))
    if not index.shape:  # If it's a single integer -> repeat for batch
      index = jnp.repeat(index, len(data_index))
    data_keys = _make_key(data_index, seed)
    batch_keys = batch_fold_in(data_keys, index)
    samples = batch_normal(batch_keys)[:, None]
    chex.assert_equal_shape([samples, data_index])
    return samples * noise_std

  return noise_fn


def _make_scaled_gaussian_index_noise(
    index_dim: int,
    noise_std: float,
    seed: int) -> NoiseFn:
  """Factory method to add Gaussian noise for index MLP."""
  std_gauss = lambda x: jax.random.normal(x, [index_dim])
  sample_std_gaussian = jax.vmap(std_gauss)

  def noise_fn(data_index: base.DataIndex,
               index: base.Index) -> chex.Array:
    """Assumes scaled Gaussian index with reserved first component."""
    chex.assert_shape(data_index, (None, 1))
    b_keys = _make_key(data_index, seed)
    b = sample_std_gaussian(b_keys)

    # Expanding the index to match the batch
    batch_size = data_index.shape[0]
    z = jnp.repeat(jnp.expand_dims(index, 0), batch_size, axis=0)
    chex.assert_shape(z, [batch_size, index_dim])
    noise = jnp.sum(b * z, axis=1, keepdims=True) * noise_std
    chex.assert_equal_shape([noise, data_index])
    return noise

  return noise_fn


def _make_gaussian_index_noise(
    index_dim: int,
    noise_std: float,
    seed: int,
) -> NoiseFn:
  """Factory method for Gaussian indexer."""
  def sample_sphere(key: chex.PRNGKey) -> chex.Array:
    x = jax.random.normal(key, shape=[index_dim])
    return x / jnp.sqrt(jnp.sum(x ** 2))
  batch_sample_sphere = jax.vmap(sample_sphere)

  def noise_fn(data_index: base.DataIndex,
               index: base.Index) -> chex.Array:
    """Assumes scaled Gaussian index with reserved first component."""
    chex.assert_shape(data_index, (None, 1))
    b_keys = _make_key(data_index, seed)
    b = batch_sample_sphere(b_keys)

    # Expanding the index to match the batch
    batch_size = data_index.shape[0]
    z = jnp.repeat(jnp.expand_dims(index, 0), batch_size, axis=0)
    chex.assert_shape(z, [batch_size, index_dim])
    noise = jnp.sum(b * z, axis=1, keepdims=True) * noise_std
    chex.assert_equal_shape([noise, data_index])
    return noise

  return noise_fn
