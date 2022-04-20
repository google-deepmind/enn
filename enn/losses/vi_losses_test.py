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
from enn.losses import vi_losses
import numpy as np


class LossesTest(parameterized.TestCase):

  @parameterized.parameters([
      [1., 0., 1.0],
      [1., 0., 10.0],
      [10., 1., 1.0],
      [10., 1., 10.0],
      [10., 10., 1.0],
  ])
  def test_analytical_diagonal_linear_model_prior_kl_fn(
      self, sigma: float, mu: float, sigma_0: float):
    """Tests the elbo function for the case of simple weights and biases."""

    num_params = 10
    kl_fn = vi_losses.get_analytical_diagonal_linear_model_prior_kl_fn(
        num_samples=1, sigma_0=sigma_0)

    w_scale = np.log(np.exp(sigma) - 1)  # sigma = log(1 + exp(w))
    params = {'layer': {
        'w': w_scale * np.ones((num_params,)),
        'b': mu * np.ones((num_params,))}}
    kl = 0.5 * num_params * (
        (sigma / sigma_0)**2
        + (mu / sigma_0)**2 - 1
        - 2 * np.log(sigma / sigma_0))

    kl_estimate = kl_fn(out=np.zeros((1, 2)),
                        params=params,
                        index=np.zeros((1, 2)))
    kl = float(np.round(kl, 2))
    kl_estimate = float(np.round(kl_estimate, 2))
    self.assertBetween((kl_estimate - kl) / (kl + 1e-9), -1e-3, 1e-3,
                       f'prior KL estimate is {kl_estimate}, expected: {kl}')


if __name__ == '__main__':
  absltest.main()
