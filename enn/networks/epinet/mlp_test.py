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
"""Tests for Epinet ENN."""

from typing import Sequence

from absl.testing import absltest
from absl.testing import parameterized
from enn import supervised
from enn import utils
from enn.networks.epinet import mlp


class EpinetTest(parameterized.TestCase):

  @parameterized.product(
      base_hiddens=[[], [10, 10]],
      epinet_hiddens=[[], [10, 10]],
      index_dim=[1, 3],
      regression=[True, False]
  )
  def test_mlp_epinet(self,
                      base_hiddens: Sequence[int],
                      epinet_hiddens: Sequence[int],
                      index_dim: int,
                      regression: bool):
    """Test that the MLP epinet runs."""
    test_experiment = supervised.make_test_experiment(regression)

    enn = mlp.make_mlp_epinet(
        output_sizes=list(base_hiddens) + [test_experiment.num_outputs,],
        epinet_hiddens=epinet_hiddens,
        index_dim=index_dim,
    )
    enn = utils.wrap_enn_with_state_as_enn(enn)
    experiment = test_experiment.experiment_ctor(enn)
    experiment.train(10)


if __name__ == '__main__':
  absltest.main()
