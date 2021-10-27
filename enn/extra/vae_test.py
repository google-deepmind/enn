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
import chex
from enn import utils
from enn.extra import vae


class VaeTest(parameterized.TestCase):

  @parameterized.parameters(
      itertools.product([True, False], [1, 2, 3]))
  def test_vae_outputs(self, bernoulli_decoder: bool, latent_dim: int):
    """Train a VAE on a test dataset and test encoder decoder functions."""
    dataset = utils.make_test_data(10)
    data = next(dataset)
    num_train, input_dim = data.x.shape

    config = vae.MLPVAEConfig(hidden_sizes=[5, 2],
                              latent_dim=latent_dim,
                              bernoulli_decoder=bernoulli_decoder,
                              num_batches=100,
                              batch_size=10)
    trained_vae = vae.get_mlp_vae_encoder_decoder(
        data_x=data.x, config=config)

    encoded_data = trained_vae.encoder(data.x)
    chex.assert_shape([encoded_data.mean, encoded_data.log_var],
                      (num_train, config.latent_dim))

    reconstructed_data = trained_vae.decoder(encoded_data.mean)
    chex.assert_shape([reconstructed_data.mean, reconstructed_data.log_var],
                      (num_train, input_dim))

if __name__ == '__main__':
  absltest.main()
