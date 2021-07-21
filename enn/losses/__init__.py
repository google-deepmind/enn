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
from enn.losses.categorical_regression import Cat2HotRegression
from enn.losses.categorical_regression import transform_to_2hot

# Prior losses
from enn.losses.prior_losses import ClassificationPriorLoss
from enn.losses.prior_losses import generate_batched_forward_at_data
from enn.losses.prior_losses import make_gaussian_dataset
from enn.losses.prior_losses import MatchingGaussianData
from enn.losses.prior_losses import RegressionPriorLoss

# Single Index
from enn.losses.single_index import AccuracyErrorLoss
from enn.losses.single_index import add_data_noise
from enn.losses.single_index import average_single_index_loss
from enn.losses.single_index import ElboLoss
from enn.losses.single_index import L2Loss
from enn.losses.single_index import SingleIndexLossFn
from enn.losses.single_index import XentLoss

# Utils
from enn.losses.utils import add_l2_weight_decay
from enn.losses.utils import combine_losses
from enn.losses.utils import combine_losses_as_metric
from enn.losses.utils import combine_single_index_losses_as_metric
from enn.losses.utils import CombineLossConfig
from enn.losses.utils import l2_weights_with_predicate

# VI losses
from enn.losses.vi_losses import get_awgn_loglike_fn
from enn.losses.vi_losses import get_categorical_loglike_fn
from enn.losses.vi_losses import get_diagonal_linear_hypermodel_elbo_fn
from enn.losses.vi_losses import get_hyperflow_elbo_fn
from enn.losses.vi_losses import get_lhm_log_model_prob_fn
from enn.losses.vi_losses import get_linear_hypermodel_elbo_fn
from enn.losses.vi_losses import get_nn_params_log_prior_prob_fn
