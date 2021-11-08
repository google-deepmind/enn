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

"""Tests for ENN Networks."""
from typing import List

from absl.testing import absltest
from absl.testing import parameterized
from enn import supervised
from enn.networks import ensembles
import haiku as hk
import jax
import numpy as np


class EinsumEnsemblesENNTest(parameterized.TestCase):

  @parameterized.product(
      input_dim=[1, 2],
      output_dim=[1, 3],
      num_ensemble=[1, 5],
  )
  def test_einsum_ensemble_enn(
      self,
      input_dim: int,
      output_dim: int,
      num_ensemble: int,
  ):
    """Simple test to run just 10 batches."""
    seed = 0

    rng = hk.PRNGSequence(seed)
    def model(inputs):
      return hk.nets.MLP([output_dim])(inputs)
    model = hk.without_apply_rng(hk.transform(model))
    enn = ensembles.EinsumEnsembleEnn(model, num_ensemble)
    params = enn.init(next(rng), np.zeros((input_dim,)), 0)
    self.assertEqual(params['mlp/~/linear_0']['b'].shape,
                     (num_ensemble, output_dim))
    self.assertEqual(params['mlp/~/linear_0']['w'].shape,
                     (num_ensemble, input_dim, output_dim))

    # overwrite random params
    params = jax.tree_map(lambda p: np.ones_like(p), params)  # pylint: disable=[unnecessary-lambda]
    dummy_inputs = np.ones(shape=(1, input_dim), dtype=np.float32)
    expected_output = (1 + input_dim) * np.ones(shape=(1, output_dim),
                                                dtype=np.float32)
    for index in range(num_ensemble):
      output = enn.apply(params, dummy_inputs, index)
      self.assertTrue(
          np.all(output - expected_output == 0),
          f'Output: {output} \n is not equal to expected: {expected_output}')


class EnsemblesTest(parameterized.TestCase):

  @parameterized.parameters([
      ([], 1, True), ([10, 10], 5, True), ([], 1, False), ([10, 10], 5, False),
  ])
  def test_ensemble(self,
                    hiddens: List[int],
                    num_ensemble: int,
                    regression: bool):
    """Simple test to run just 10 batches."""
    test_experiment = supervised.make_test_experiment(regression)

    enn = ensembles.MLPEnsembleMatchedPrior(
        output_sizes=hiddens+[test_experiment.num_outputs],
        dummy_input=test_experiment.dummy_input,
        num_ensemble=num_ensemble,
        prior_scale=1.,
    )
    experiment = test_experiment.experiment_ctor(enn)
    experiment.train(10)


if __name__ == '__main__':
  absltest.main()
