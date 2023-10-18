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

"""Tests for ENN Networks."""

from typing import List

from absl.testing import absltest
from absl.testing import parameterized
from enn import supervised
from enn.networks import einsum_mlp

from jax import config
config.update('jax_numpy_rank_promotion', 'raise')


class EinsumMlpTest(parameterized.TestCase):

  @parameterized.parameters([
      ([], 1, True), ([10, 10], 5, True), ([], 1, False), ([10, 10], 5, False),
  ])
  def test_ensemble(self,
                    hiddens: List[int],
                    num_ensemble: int,
                    regression: bool):
    """Simple test to run just 10 batches."""
    test_experiment = supervised.make_test_experiment(regression)

    enn = einsum_mlp.make_ensemble_mlp_with_prior_enn(
        output_sizes=hiddens+[test_experiment.num_outputs],
        dummy_input=test_experiment.dummy_input,
        num_ensemble=num_ensemble,
        prior_scale=1.,
    )
    experiment = test_experiment.experiment_ctor(enn)
    experiment.train(10)


if __name__ == '__main__':
  absltest.main()
