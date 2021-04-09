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

"""Tests for enn.networks.categorical_ensembles."""
from typing import List

from absl.testing import absltest
from absl.testing import parameterized
from enn import base
from enn import losses
from enn import supervised
from enn import utils
from enn.networks import categorical_ensembles
import numpy as np
import optax


class CategoricalEnsemblesTest(parameterized.TestCase):

  @parameterized.parameters([
      [[20], np.linspace(-5, 5, 10), 3],
      [[], np.linspace(-1, 1, 10), 1],
  ])
  def test_categorical_ensemble(self,
                                hiddens: List[int],
                                atoms: base.Array,
                                num_ensemble: int):
    """Running with the naive L2 loss."""
    test_experiment = supervised.make_test_experiment(regression=True)
    enn = categorical_ensembles.CatMLPEnsembleGpPrior(
        output_sizes=hiddens + [1],
        atoms=atoms,
        input_dim=test_experiment.dummy_input.shape[1],
        num_ensemble=num_ensemble,
        num_feat=10,
    )
    experiment = test_experiment.experiment_ctor(enn)
    experiment.train(10)

  def test_categorical_2hot_regression(self):
    """Running with the categorical regression loss."""
    dataset = utils.make_test_data()
    enn = categorical_ensembles.CatMLPEnsembleMlpPrior(
        output_sizes=[50, 50, 1],
        atoms=np.linspace(-1, 1, 10),
        dummy_input=next(dataset)['x'],
        num_ensemble=3,
    )
    single_loss = losses.Cat2HotRegressionWithBootstrap()
    loss_fn = losses.average_single_index_loss(single_loss, 1)
    experiment = supervised.Experiment(enn, loss_fn, optax.adam(1e-3), dataset)
    experiment.train(10)


if __name__ == '__main__':
  absltest.main()
