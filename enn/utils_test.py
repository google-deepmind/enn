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
"""Tests for enn.utils."""
from absl.testing import absltest
from absl.testing import parameterized
from enn import base
from enn import networks
from enn import utils
import jax
import jax.numpy as jnp


class UtilsTest(parameterized.TestCase):

  @parameterized.parameters([
      [networks.PrngIndexer(), 23],
      [networks.ScaledGaussianIndexer(10), 32],
      [networks.GaussianWithUnitIndexer(5), 40],
      [networks.EnsembleIndexer(13), 50],
  ])
  def test_batch_indexer(
      self, indexer: base.EpistemicIndexer, batch_size: int):
    batch_indexer = utils.make_batch_indexer(indexer, batch_size)
    batch_index = batch_indexer(jax.random.PRNGKey(0))
    # Check that the batch index is of the right leading dimension
    assert batch_index.shape[0] == batch_size
    # Check that they are not all identical
    assert not jnp.isclose(batch_index, batch_index[0]).all()


if __name__ == '__main__':
  absltest.main()
