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

"""Exposing the public methods of the losses."""
# Categorical regression
from enn.losses.categorical_regression import Cat2HotRegressionWithBootstrap
from enn.losses.categorical_regression import transform_to_2hot

# Prior losses
from enn.losses.prior_losses import ClassificationPriorLoss
from enn.losses.prior_losses import MatchingGaussianData
from enn.losses.prior_losses import RegressionPriorLoss
from enn.losses.prior_losses import SpecialRegressionData

# Single Index
from enn.losses.single_index import AccuracyErrorLoss
from enn.losses.single_index import average_single_index_loss
from enn.losses.single_index import ElboLoss
from enn.losses.single_index import L2LossWithBootstrap
from enn.losses.single_index import SingleIndexLossFn
from enn.losses.single_index import XentLossWithBootstrap

# Utils
from enn.losses.utils import add_l2_weight_decay
from enn.losses.utils import combine_losses_as_metric
from enn.losses.utils import combine_single_index_losses_as_metric
from enn.losses.utils import l2_weights_excluding_name
