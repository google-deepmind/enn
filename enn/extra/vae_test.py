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

"""Tests for VAE."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
from enn import losses
from enn import supervised
from enn import utils
from enn.extra import vae
import jax
import optax


class VaeTest(parameterized.TestCase):

  @parameterized.parameters(
      itertools.product([True, False], [0, 100]))
  def test_train_decreases_loss(self, bernoulli_decoder: bool, seed: int):
    """Train a VAE on a test dataset and make sure loss decreases."""
    dataset = utils.make_test_data(100)
    data = next(dataset)
    input_size = data.x.shape[-1]
    enn = vae.MLPVae(input_size)

    log_likelihood_fn = losses.get_log_likelihood_fn(bernoulli_decoder)
    latent_kl_fn = losses.get_latent_kl_fn()
    single_loss = losses.VaeLoss(log_likelihood_fn, latent_kl_fn)
    loss_fn = losses.average_single_index_loss(single_loss, num_index_samples=1)

    optimizer = optax.adam(1e-3)

    experiment = supervised.Experiment(enn, loss_fn, optimizer, dataset)

    init_key, loss_key = jax.random.split(jax.random.PRNGKey(seed), 2)
    initial_loss = experiment.loss(data, init_key)
    experiment.train(num_batches=50)
    final_loss = experiment.loss(data, loss_key)
    self.assertGreater(
        initial_loss, final_loss,
        f'final loss {final_loss} is greater than initial loss {initial_loss}')


if __name__ == '__main__':
  absltest.main()
