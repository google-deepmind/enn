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

"""Tests for ENN Networks."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
from enn.extra import kmeans
import haiku as hk
import jax
import jax.numpy as jnp


class KmeansTest(parameterized.TestCase):

  @parameterized.product(
      num_x=[10, 100],
      dim_x=[1, 5],
      num_centroids=[1, 3],
  )
  def test_kmeans_runs(self, num_x: int, dim_x: int, num_centroids: int):
    """Test that KMeans clustering runs and has no nan."""
    rng = hk.PRNGSequence(999)
    x = jax.random.normal(next(rng), [num_x, dim_x])
    kmeans_cluster = kmeans.KMeansCluster(
        num_centroids=num_centroids,
        num_iterations=100,
        key=next(rng),
    )
    output = kmeans_cluster.fit(x)

    chex.assert_shape(output.centroids, [num_centroids, dim_x])
    chex.assert_shape(output.counts_per_centroid, [num_centroids])
    chex.assert_shape(output.std_distance, [num_centroids])
    assert jnp.all(jnp.isfinite(output.centroids))
    assert jnp.all(jnp.isfinite(output.counts_per_centroid))
    assert jnp.all(jnp.isfinite(output.std_distance))


if __name__ == '__main__':
  absltest.main()
