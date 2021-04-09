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

"""Tests for ENN Hypermodels."""
from typing import List

from absl.testing import absltest
from absl.testing import parameterized
from enn import supervised
from enn.networks import hypermodels
from enn.networks import indexers
import haiku as hk


class MLPHypermodelTest(parameterized.TestCase):

  @parameterized.parameters([
      ([], [], 4, True),
      ([3], [], 5, True),
      ([3, 7], [4], 3, True),
      ([], [], 4, False),
      ([3], [], 5, False),
      ([3, 7], [4], 3, False),
  ])
  def test_ten_batches(self, model_hiddens: List[int], hyper_hiddens: List[int],
                       index_dim: int, regression: bool):
    """Simple test to run just 10 batches."""
    test_experiment = supervised.make_test_experiment(regression)

    def base_net(x):
      return hk.nets.MLP(model_hiddens + [test_experiment.num_outputs])(x)

    transformed_base = hk.without_apply_rng(hk.transform(base_net))

    indexer = indexers.ScaledGaussianIndexer(index_dim, index_scale=1.0)
    enn = hypermodels.MLPHypermodel(
        transformed_base=transformed_base,
        dummy_input=test_experiment.dummy_input,
        indexer=indexer,
        hidden_sizes=hyper_hiddens,
    )
    experiment = test_experiment.experiment_ctor(enn)
    experiment.train(10)

  @parameterized.parameters([
      ([], [], [], [], 0.0, 4, True),
      ([3], [], [4], [], 1.0, 4, True),
      ([3, 7], [4], [4], [], 1.0, 4, True),
      ([3, 7], [4], [4, 6], [5], 1.0, 4, True),
      ([], [], [], [], 0.0, 4, False),
      ([3], [], [4], [], 1.0, 4, False),
      ([3, 7], [4], [4], [], 1.0, 4, False),
      ([3, 7], [4], [4, 6], [5], 1.0, 4, False),
  ])
  def test_hyper_prior(self, model_hiddens: List[int], hyper_hiddens: List[int],
                       prior_model_hiddens: List[int],
                       prior_hyper_hiddens: List[int], prior_scale: float,
                       index_dim: int, regression: bool):
    """Simple test to run just 10 batches."""
    test_experiment = supervised.make_test_experiment(regression)

    indexer = indexers.ScaledGaussianIndexer(index_dim, index_scale=1.0)
    enn = hypermodels.MLPHypermodelWithHypermodelPrior(
        base_output_sizes=model_hiddens + [test_experiment.num_outputs],
        prior_scale=prior_scale,
        dummy_input=test_experiment.dummy_input,
        indexer=indexer,
        prior_base_output_sizes=prior_model_hiddens +
        [test_experiment.num_outputs],
        hyper_hidden_sizes=hyper_hiddens,
        prior_hyper_hidden_sizes=prior_hyper_hiddens)

    experiment = test_experiment.experiment_ctor(enn)
    experiment.train(10)


if __name__ == '__main__':
  absltest.main()
