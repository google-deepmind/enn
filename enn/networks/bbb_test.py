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

"""Tests for ENN Hypermodels."""
from typing import List

from absl.testing import absltest
from absl.testing import parameterized
from enn import supervised
from enn.networks import bbb


class BBBTest(parameterized.TestCase):
  @parameterized.parameters([
      ([3, 7], True),
      ([3, 7], False),
  ])

  def test_bbb(
      self, model_hiddens: List[int], regression: bool):
    """Simple test to run just 10 batches."""
    test_experiment = supervised.make_test_experiment(regression)

    enn = bbb.make_bbb_enn(
        base_output_sizes=model_hiddens + [test_experiment.num_outputs],
        dummy_input=test_experiment.dummy_input)
    experiment = test_experiment.experiment_ctor(enn)
    experiment.train(10)


if __name__ == '__main__':
  absltest.main()
