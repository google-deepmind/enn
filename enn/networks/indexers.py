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

"""Epistemic indexers for ENNs."""
import dataclasses
from enn import base
import jax
import jax.numpy as jnp


class PrngIndexer(base.EpistemicIndexer):
  """Index by JAX PRNG sequence."""

  def __call__(self, key: base.RngKey) -> base.Index:
    return key


@dataclasses.dataclass
class EnsembleIndexer(base.EpistemicIndexer):
  """Index into an ensemble by integer."""
  num_ensemble: int

  def __call__(self, key: base.RngKey) -> base.Index:
    return jax.random.randint(key, [], 0, self.num_ensemble)


@dataclasses.dataclass
class GaussianIndexer(base.EpistemicIndexer):
  """A Gaussian indexer ~ N(0, I)."""
  index_dim: int

  def __call__(self, key: base.RngKey) -> base.Index:
    return jax.random.normal(key, shape=[self.index_dim])


@dataclasses.dataclass
class ScaledGaussianIndexer(base.EpistemicIndexer):
  """A scaled Gaussian indexer."""
  index_dim: int
  # When index_scale is 1.0 the returned random variable has expected norm = 1.
  index_scale: float = 1.0

  def __call__(self, key: base.RngKey) -> base.Index:
    return self.index_scale / jnp.sqrt(self.index_dim) * jax.random.normal(
        key, shape=[self.index_dim])


@dataclasses.dataclass
class GaussianWithUnitIndexer(base.EpistemicIndexer):
  """Produces index (1, z) for z dimension=index_dim-1 unit ball."""
  index_dim: int

  @property
  def mean_index(self) -> base.Array:
    return jnp.append(1, jnp.zeros(self.index_dim - 1))

  def __call__(self, key: base.RngKey) -> base.Index:
    return jnp.append(1, jax.random.normal(
        key, shape=[self.index_dim - 1]) / jnp.sqrt(self.index_dim - 1))


@dataclasses.dataclass
class DirichletIndexer(base.EpistemicIndexer):
  """Samples a Dirichlet index with parameter alpha."""
  alpha: base.Array

  def __call__(self, key: base.RngKey) -> base.Index:
    return jax.random.dirichlet(key, self.alpha)


@dataclasses.dataclass
class BatchNormIndexer(base.EpistemicIndexer):
  """Returns a tuple (indexer(key), is_training) for use in batchnorm nets."""
  indexer: base.EpistemicIndexer
  is_training: bool

  def __call__(self, key: base.RngKey) -> base.Index:
    """Returns a tuple (indexer(key), is_training) for use in batchnorm nets."""
    return (self.indexer(key), self.is_training)
