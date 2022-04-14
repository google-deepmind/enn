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
from enn.supervised import multiloss_experiment
import jax
import optax


class ExperimentTest(parameterized.TestCase):

  @parameterized.parameters(itertools.product([1, 3], [0, 1, 2]))
  def test_train_decreases_loss(self, num_classes: int, seed: int):
    """Train an ensemble ENN on a test dataset and make sure loss decreases."""
    # Creat ENN and loss functions

    if num_classes == 1:
      single_loss = losses.L2Loss()
    else:
      single_loss = losses.XentLoss(num_classes)
    loss_fn = losses.average_single_index_loss(single_loss, 2)

    # Create two different training losses
    train_dataset = utils.make_test_data(30)
    base_trainer = multiloss_experiment.MultilossTrainer(
        loss_fn=loss_fn,
        dataset=train_dataset,
        should_train=lambda _: True,
    )
    prior_dataset = utils.make_test_data(2)  # An example of alternative data
    prior_trainer = multiloss_experiment.MultilossTrainer(
        loss_fn=loss_fn,
        dataset=prior_dataset,
        should_train=lambda step: step % 2 == 0,
        name='prior'
    )

    enn = networks.MLPEnsembleMatchedPrior(
        output_sizes=[20, 20, num_classes],
        num_ensemble=2,
        dummy_input=next(train_dataset).x,
    )

    experiment = multiloss_experiment.MultilossExperiment(
        enn=enn,
        trainers=[base_trainer, prior_trainer],
        optimizer=optax.adam(1e-3),
        seed=seed,
    )
    init_key, loss_key = jax.random.split(jax.random.PRNGKey(seed), 2)
    initial_loss = experiment.loss(next(train_dataset), init_key)
    experiment.train(50)
    final_loss = experiment.loss(next(train_dataset), loss_key)
    self.assertGreater(
        initial_loss, final_loss,
        f'final loss {final_loss} is greater than initial loss {initial_loss}')


if __name__ == '__main__':
  absltest.main()
