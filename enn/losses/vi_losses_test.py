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
from enn import base as enn_base
from enn.losses import vi_losses
import numpy as np


class LossesTest(parameterized.TestCase):

  @parameterized.parameters([
      [1., 0., 1.0],
      [1., 0., 10.0],
      [10., 1., 1.0],
      [10., 10., 1.0],
  ])
  def test_diagonal_linear_hypermodel_elbo_fn(
      self, sigma: float, mu: float, sigma_0: float):
    """Tests the elbo function for the case of simple weights and biases."""

    num_params = 10
    log_likelihood_fn = lambda *args: np.array(0.0)
    nelbo_fn = vi_losses.get_diagonal_linear_hypermodel_elbo_fn(
        log_likelihood_fn, sigma_0=sigma_0, num_samples=1)

    w_scale = np.log(np.exp(sigma) - 1)  # sigma = log(1 + exp(w))
    params = {'layer': {
        'w': w_scale * np.ones(num_params,),
        'b': mu * np.ones(num_params,)}}
    kl = 0.5 * num_params * (sigma**2 + mu**2 / sigma_0 - 1 - 2 * np.log(sigma))
    batch = enn_base.Batch(x=np.zeros((2, 1)), y=np.zeros((2, 1)))
    nelbo, _ = nelbo_fn(apply=lambda *args: np.zeros((2,)),
                        params=params,
                        batch=batch,
                        index=np.zeros((2,)))
    self.assertEqual(kl, nelbo, f'negetive elbos is {nelbo}, expected: {kl}')


if __name__ == '__main__':
  absltest.main()
