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

"""Setting up simple experiments for tests."""
from typing import Callable, NamedTuple

import chex
from enn import losses
from enn import networks
from enn import utils
from enn.supervised import sgd_experiment
import optax


class TestExperiment(NamedTuple):
  num_outputs: int
  experiment_ctor: Callable[[networks.EpistemicNetwork],
                            sgd_experiment.Experiment]
  dummy_input: chex.Array


def make_test_experiment(regression: bool) -> TestExperiment:
  """Utility function to set up a supervised experiment for testing."""
  dataset = utils.make_test_data(20)
  optimizer = optax.adam(1e-3)
  if regression:
    num_outputs = 1
    single_loss = losses.L2LossWithState()
  else:
    num_outputs = 2
    single_loss = losses.XentLossWithState(num_outputs)

  loss_fn = losses.average_single_index_loss_with_state(
      single_loss, num_index_samples=1)
  return TestExperiment(
      num_outputs=num_outputs,
      experiment_ctor=lambda enn: sgd_experiment.Experiment(  # pylint:disable=g-long-lambda
          enn, loss_fn, optimizer, dataset),
      dummy_input=next(dataset).x,
  )
