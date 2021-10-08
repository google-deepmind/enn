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
from enn import utils
from enn.extra import vae
import jax


class VaeTest(parameterized.TestCase):

  @parameterized.parameters(
      itertools.product([True, False], [0, 100]))
  def test_train_decreases_loss(self, bernoulli_decoder: bool, seed: int):
    """Train a VAE on a test dataset and make sure loss decreases."""
    dataset = utils.make_test_data(100)
    data = next(dataset)

    vae_trainer = vae.VaeTrainer(
        data,
        hidden_sizes=(8, 4),
        latent_size=2,
        bernoulli_decoder=bernoulli_decoder,
        batch_size=100,
        seed=seed)
    init_key, loss_key = jax.random.split(jax.random.PRNGKey(seed), 2)
    initial_loss = vae_trainer.loss(data.x, init_key)
    vae_trainer.train(num_batches=50)
    final_loss = vae_trainer.loss(data.x, loss_key)
    self.assertGreater(
        initial_loss, final_loss,
        f'final loss {final_loss} is greater than initial loss {initial_loss}')


if __name__ == '__main__':
  absltest.main()
