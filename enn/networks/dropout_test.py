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
from typing import Sequence
from absl.testing import absltest
from absl.testing import parameterized
from enn import supervised
from enn import utils
from enn.networks import dropout


class NetworkTest(parameterized.TestCase):

  @parameterized.product(
      hiddens=[[], [10, 10]],
      dropout_rate=[0.05, 0.2, 0.5],
      dropout_input=[True, False],
      regression=[True, False])
  def test_dropout_mlp(self, hiddens: Sequence[int], dropout_rate: float,
                       dropout_input: bool, regression: bool):
    """Simple test to run just 10 batches."""
    test_experiment = supervised.make_test_experiment(regression)
    enn = dropout.MLPDropoutENN(
        output_sizes=list(hiddens)+[test_experiment.num_outputs],
        dropout_rate=dropout_rate,
        dropout_input=dropout_input
    )
    enn = utils.wrap_enn_with_state_as_enn(enn)
    experiment = test_experiment.experiment_ctor(enn)
    experiment.train(10)


if __name__ == '__main__':
  absltest.main()
