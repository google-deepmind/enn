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
from enn.networks import resnet
import haiku as hk
import jax


class NetworkTest(parameterized.TestCase):

  @parameterized.product(
      num_classes=[2, 10],
      batch_size=[1, 10],
      image_size=[2, 10],
  )
  def test_forward_pass(
      self,
      num_classes: int,
      batch_size: int,
      image_size: int,
  ):
    """Tests forward pass and output shape."""
    enn = resnet.EnsembleResNetENN(
        num_output_classes=num_classes,
    )
    rng = hk.PRNGSequence(0)
    image_shape = [image_size, image_size, 3]
    x = jax.random.normal(next(rng), shape=[batch_size,] + image_shape)
    index = enn.indexer(next(rng))
    params, state = enn.init(next(rng), x, index)
    out, unused_new_state = enn.apply(params, state, x, index)
    self.assertEqual(out.shape, (batch_size, num_classes))


if __name__ == '__main__':
  absltest.main()
