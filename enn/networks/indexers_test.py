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

"""Tests for enn.networks.indexers."""
from absl.testing import absltest
from absl.testing import parameterized
from enn import base_legacy
from enn.networks import indexers
import jax
import numpy as np


class IndexersTest(parameterized.TestCase):

  @parameterized.parameters([
      [indexers.GaussianWithUnitIndexer(10)],
      [indexers.EnsembleIndexer(5)],
      [indexers.PrngIndexer()],
      [indexers.ScaledGaussianIndexer(7)],
      [indexers.DirichletIndexer(np.ones(3))],
  ])
  def test_index_forward(self, indexer: base_legacy.EpistemicIndexer):
    key = jax.random.PRNGKey(777)
    jit_indexer = jax.jit(lambda x: indexer(x))  # pylint: disable=unnecessary-lambda
    assert np.allclose(indexer(key), jit_indexer(key))


if __name__ == '__main__':
  absltest.main()
