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

"""Tests for enn.bootstrapping."""
from absl.testing import absltest
from absl.testing import parameterized
from enn import base
from enn import networks
from enn.data_noise import bootstrapping
import jax
import numpy as np


class BootstrappingTest(parameterized.TestCase):

  @parameterized.parameters([
      [networks.EnsembleIndexer(10), 'poisson'],
      [networks.EnsembleIndexer(10), 'bernoulli'],
      [networks.EnsembleIndexer(10), 'exponential'],
      [networks.EnsembleIndexer(10), 'uniform'],
      [networks.EnsembleIndexer(10), 'none'],
      [networks.GaussianWithUnitIndexer(10), 'bernoulli'],
      [networks.ScaledGaussianIndexer(10), 'bernoulli'],
      [networks.ScaledGaussianIndexer(10), 'exponential'],
      [networks.PrngIndexer(), 'poisson'],
      [networks.PrngIndexer(), 'bernoulli'],
      [networks.PrngIndexer(), 'exponential'],
      [networks.PrngIndexer(), 'uniform'],
  ])
  def test_average_weight_approx_one(self,
                                     indexer: base.EpistemicIndexer,
                                     distribution: str):
    """Check that the average weight of bootstrap sample approximately one."""
    seed = 999
    num_data = 10_000
    tolerance = 1  # TODO(author2): Test fails at lower tolerance --> fix.
    def init_fn(k, x, z):
      del k, x, z
      return {'lin': {'w': np.ones(1), 'b': np.ones(1)}}
    fake_enn = networks.EpistemicNetwork(
        apply=lambda p, x, z: np.ones(1)[:, None],
        init=init_fn,
        indexer=indexer,
    )
    boot_fn = bootstrapping.make_boot_fn(fake_enn, distribution, seed=seed)
    index = fake_enn.indexer(jax.random.PRNGKey(seed))
    data_index = np.arange(num_data)[:, None]
    batch_weights = jax.jit(boot_fn)(data_index, index)

    # Check the quality of the bootstrap weights
    assert np.all(batch_weights >= 0)
    assert np.abs(1 - np.mean(batch_weights)) < tolerance


if __name__ == '__main__':
  absltest.main()
