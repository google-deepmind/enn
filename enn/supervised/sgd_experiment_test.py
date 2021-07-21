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

"""Tests for enn.supervised.sgd_experiment."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
from enn import losses
from enn import networks
from enn import utils
from enn.supervised.sgd_experiment import Experiment
import jax
import optax


class ExperimentTest(parameterized.TestCase):

  @parameterized.parameters(itertools.product([1, 3], [0, 1]))
  def test_train_decreases_loss(self, num_outputs: int, seed: int):
    """Train an ensemble ENN on a test dataset and make sure loss decreases."""

    num_ensemble = 5
    output_sizes = [8, 8, num_outputs]
    enn = networks.MLPEnsembleEnn(
        output_sizes=output_sizes,
        num_ensemble=num_ensemble,
    )

    dataset = utils.make_test_data(100)
    optimizer = optax.adam(1e-3)
    if num_outputs == 1:
      single_loss = losses.L2Loss()
    elif num_outputs > 1:
      single_loss = losses.XentLoss(num_outputs)
    else:
      raise ValueError(f'num_outputs should be >= 1. It is {num_outputs}.')
    loss_fn = losses.average_single_index_loss(single_loss,
                                               num_index_samples=10)
    experiment = Experiment(enn, loss_fn, optimizer, dataset, seed)
    init_key, loss_key = jax.random.split(jax.random.PRNGKey(seed), 2)
    initial_loss = experiment.loss(next(dataset), init_key)
    experiment.train(50)
    final_loss = experiment.loss(next(dataset), loss_key)
    self.assertGreater(
        initial_loss, final_loss,
        f'final loss {final_loss} is greater than initial loss {initial_loss}')


if __name__ == '__main__':
  absltest.main()
