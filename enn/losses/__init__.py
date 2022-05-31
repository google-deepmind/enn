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
from enn.losses.single_index import SingleIndexLossFnBase
from enn.losses.single_index import VaeLoss
from enn.losses.single_index import XentLoss

# Single Index With State
from enn.losses.single_index_with_state import add_data_noise_to_loss_with_state
from enn.losses.single_index_with_state import average_single_index_loss_with_state
from enn.losses.single_index_with_state import SingleIndexLossFnWithState
from enn.losses.single_index_with_state import SingleIndexLossFnWithStateBase
from enn.losses.single_index_with_state import xent_loss_with_state_custom_labels
from enn.losses.single_index_with_state import XentLossWithState

# Utils
from enn.losses.utils import add_l2_weight_decay
from enn.losses.utils import combine_losses
from enn.losses.utils import combine_losses_as_metric
from enn.losses.utils import combine_single_index_losses_as_metric
from enn.losses.utils import CombineLossConfig
from enn.losses.utils import CombineLossConfigBase
from enn.losses.utils import l2_weights_with_predicate
from enn.losses.utils import PredicateFn

# VAE losses
from enn.losses.vae_losses import binary_log_likelihood
from enn.losses.vae_losses import gaussian_log_likelihood
from enn.losses.vae_losses import get_log_likelihood_fn
from enn.losses.vae_losses import latent_kl_divergence
from enn.losses.vae_losses import latent_kl_fn
from enn.losses.vae_losses import LogLikelihoodFn

# VI losses
from enn.losses.vi_losses import get_analytical_diagonal_linear_model_prior_kl_fn
from enn.losses.vi_losses import get_analytical_hyperflow_model_prior_kl_fn
from enn.losses.vi_losses import get_analytical_linear_model_prior_kl_fn
from enn.losses.vi_losses import get_awgn_loglike_fn
from enn.losses.vi_losses import get_categorical_loglike_fn
from enn.losses.vi_losses import get_sample_based_model_prior_kl_fn
