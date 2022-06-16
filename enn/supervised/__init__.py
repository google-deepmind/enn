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

"""Exposing the public methods of the supervised experiments."""

# Base
from enn.supervised.base import BaseExperiment

# Multiloss
from enn.supervised.multiloss_experiment import MultilossExperiment
from enn.supervised.multiloss_experiment import MultilossTrainer
from enn.supervised.multiloss_experiment import TrainingState

# Multiloss legacy
from enn.supervised.multiloss_experiment_legacy import MultilossExperiment as MultilossExperimentLegacy
from enn.supervised.multiloss_experiment_legacy import MultilossTrainer as MultilossTrainerLegacy
from enn.supervised.multiloss_experiment_legacy import TrainingState as TrainingStateLegacy

# Experiments
from enn.supervised.sgd_experiment import Experiment

# Experiments
from enn.supervised.sgd_experiment_legacy import Experiment as ExperimentLegacy

# Testing
from enn.supervised.testing import make_test_experiment
from enn.supervised.testing import TestExperiment
