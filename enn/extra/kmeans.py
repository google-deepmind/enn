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

"""Jax implementation of KMeans clustering."""

import dataclasses
from typing import NamedTuple, Tuple

import chex
import jax
import jax.numpy as jnp


class KMeansOutput(NamedTuple):
  centroids: chex.Array  # Centroids found by algorithm: [num_centoid, dim_x]
  counts_per_centroid: chex.Array  # Counts per centroid: [num_centroid]
  std_distance: chex.Array  # Std distance to centroid: [num_centroid]
  classes: chex.Array  # Cluster index of data: [num_data_samples]


@dataclasses.dataclass
class KMeansCluster:
  """Performs KMeans clustering on data."""
  num_centroids: int
  num_iterations: int
  key: chex.PRNGKey

  def fit(self, x: chex.Array) -> KMeansOutput:
    """Fits KMeans cluster to given data."""
    # Initialize centroids randomly
    random_idx = jax.random.choice(
        self.key, x.shape[0], [self.num_centroids], replace=False)
    initial_centroids = x[random_idx, :]
    initial_state = _TrainingState(initial_centroids, iter=0)

    # Perfom KMeans via jax.lax.while_loop
    cond_fn = lambda state: state.iter < self.num_iterations
    body_fn = lambda state: kmeans_iteration(x, state)
    final_state = jax.lax.while_loop(cond_fn, body_fn, initial_state)

    return jax.jit(compute_output)(x, final_state)


class _TrainingState(NamedTuple):
  centroids: chex.Array  # Centroids: [num_centroid, dim_x]
  iter: int  # Training iteration


def get_classes_and_distances(
    x: chex.Array, centroids: chex.Array) -> Tuple[chex.Array, chex.Array]:
  """Assigns x to nearest centroid and computes distance to each centroid."""
  chex.assert_rank([x, centroids], 2)
  num_x, dim_x = x.shape
  num_centroids, dim_centroids = centroids.shape
  chex.assert_equal(dim_x, dim_centroids)
  norm_per_x_per_class = jax.vmap(jax.vmap(jnp.linalg.norm))
  distances = norm_per_x_per_class(
      jnp.expand_dims(centroids, 0) - jnp.expand_dims(x, 1))
  chex.assert_shape(distances, (num_x, num_centroids))
  classes = jnp.argmin(distances, axis=1)
  chex.assert_shape(classes, [num_x])
  return classes, distances


def _safe_divide(numerator: chex.Array, denominator: chex.Array) -> chex.Array:
  safe_denom = jnp.maximum(denominator, 1e-6)
  return numerator / safe_denom


def kmeans_iteration(x: chex.Array, state: _TrainingState) -> _TrainingState:
  """Performs one iteration of kmeans clustering."""
  num_x, dim_x = x.shape
  num_centroids = state.centroids.shape[0]

  # Form one-hot masks
  classes, _ = get_classes_and_distances(x, centroids=state.centroids)
  one_hot_centroids = jax.nn.one_hot(classes, num_centroids)
  chex.assert_shape(one_hot_centroids, [num_x, num_centroids])

  # Take mean over classes for new centroids.
  masked_x = x[:, None, :] * one_hot_centroids[:, :, None]
  chex.assert_shape(masked_x, [num_x, num_centroids, dim_x])
  sum_per_centroid = jnp.sum(masked_x, axis=0)
  count_per_centroid = jnp.sum(one_hot_centroids, axis=0)
  new_centroids = _safe_divide(sum_per_centroid, count_per_centroid[:, None])
  chex.assert_shape(new_centroids, [num_centroids, dim_x])

  return _TrainingState(new_centroids, state.iter + 1)


def compute_output(x: chex.Array, state: _TrainingState) -> KMeansOutput:
  """Parse the final output, which includes std per class."""
  # Pulling out shapes
  num_centroids = state.centroids.shape[0]

  # Computing distances
  classes, distances = get_classes_and_distances(x, state.centroids)
  one_hot_centroids = jax.nn.one_hot(classes, num_centroids)
  chex.assert_equal_shape([distances, one_hot_centroids])

  # Std per class
  counts_per_centroid = jnp.sum(one_hot_centroids, axis=0)
  masked_sq_distances = jnp.square(one_hot_centroids * distances)
  total_sq_distance_per_class = jnp.sum(masked_sq_distances, axis=0)
  std_distance = _safe_divide(total_sq_distance_per_class, counts_per_centroid)

  return KMeansOutput(state.centroids, counts_per_centroid,
                      std_distance, classes)

