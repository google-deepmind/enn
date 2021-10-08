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
"""Tests for ENN vi losses."""

from absl.testing import absltest
from absl.testing import parameterized
from enn.extra import vae_losses
import numpy as np


class LossesTest(parameterized.TestCase):

  @parameterized.product(input_size=[1, 10, 100],
                         batch_size=[10])
  def test_binary_log_likelihood(self, input_size: int, batch_size: int):
    """Tests the binary log likelihood."""

    x = np.zeros((batch_size, input_size))
    output = np.zeros_like(x)

    log_likelihood = vae_losses.binary_log_likelihood(x, output)

    result = -1 * input_size * np.log(2) * np.ones((batch_size,))
    np.testing.assert_almost_equal(
        log_likelihood,
        result,
        decimal=3,
        err_msg=f'log_likelihood is {log_likelihood}, expected: {result}')

  @parameterized.product(input_size=[1, 10, 100], batch_size=[10])
  def test_gaussian_log_likelihood(self, input_size: int, batch_size: int):
    """Tests the binary log likelihood."""

    x = np.zeros((batch_size, input_size))
    mean = np.zeros_like(x)
    log_variance = np.zeros_like(x)

    log_likelihood = vae_losses.gaussian_log_likelihood(x, mean, log_variance)

    result = -0.5 * input_size * np.log(2 * np.pi) * np.ones((batch_size,))
    np.testing.assert_almost_equal(
        log_likelihood,
        result,
        decimal=3,
        err_msg=f'log_likelihood is {log_likelihood}, expected: {result}')

  @parameterized.product(input_size=[1, 10, 100], batch_size=[10])
  def test_latent_kl(self, input_size: int, batch_size: int):
    """Tests the binary log likelihood."""

    mean = np.zeros((batch_size, input_size))
    log_variance = np.zeros_like(mean)

    log_likelihood = vae_losses.latent_kl_divergence(mean, log_variance)

    result = 0 * np.ones((batch_size,))
    np.testing.assert_almost_equal(
        log_likelihood,
        result,
        decimal=3,
        err_msg=f'log_likelihood is {log_likelihood}, expected: {result}')


if __name__ == '__main__':
  absltest.main()
