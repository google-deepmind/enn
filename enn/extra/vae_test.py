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
  def test_vae_outputs(self, bernoulli_decoder: bool, latent_size: int):
    """Train a VAE on a test dataset and test encoder decoder functions."""
    dataset = utils.make_test_data(100)
    data = next(dataset)
    input_size = data.x.shape[-1]

    config = vae.MLPVAEConfig(input_size=input_size,
                              hidden_sizes=[5, 2],
                              latent_size=latent_size,
                              bernoulli_decoder=bernoulli_decoder,
                              num_batches=25,
                              batch_size=10,
                              train_log_freq=5)
    encoder_fn, decoder_fn = vae.get_mlp_vae_encoder_decoder(
        config=config, data_x=data.x)

    encoded_data = encoder_fn(data.x)
    chex.assert_shape(encoded_data.mean, (data.x.shape[0], config.latent_size))
    chex.assert_shape(encoded_data.log_variance, encoded_data.mean.shape)

    reconstructed_data = decoder_fn(encoded_data.mean)
    chex.assert_shape(reconstructed_data.mean, (data.x.shape[0], input_size))
    chex.assert_shape(reconstructed_data.log_variance,
                      reconstructed_data.mean.shape)


if __name__ == '__main__':
  absltest.main()
