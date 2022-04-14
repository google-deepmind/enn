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
"""Helpful losses for the ENN agent."""

from typing import Callable, Optional

from enn import base as enn_base
from enn import data_noise
from enn import losses

from enn.experiments.neurips_2021 import base as testbed_base


EnnCtor = Callable[[testbed_base.PriorKnowledge], enn_base.EpistemicNetwork]
LossCtor = Callable[
    [testbed_base.PriorKnowledge, enn_base.EpistemicNetwork], enn_base.LossFn]


def default_enn_loss(num_index_samples: int = 10,
                     distribution: str = 'none',
                     seed: int = 0,
                     weight_reg_scale: Optional[float] = None) -> LossCtor:
  """Constructs a default loss suitable for classification or regression."""
  def loss_ctor(prior: testbed_base.PriorKnowledge,
                enn: enn_base.EpistemicNetwork) -> enn_base.LossFn:
    single_loss = losses.L2Loss()

    # Add bootstrapping
    boot_fn = data_noise.BootstrapNoise(enn, distribution, seed)
    single_loss = losses.add_data_noise(single_loss, boot_fn)

    loss_fn = losses.average_single_index_loss(single_loss, num_index_samples)

    # Add L2 weight decay
    if weight_reg_scale:
      scale = (weight_reg_scale ** 2) / (2. * prior.num_train)
      loss_fn = losses.add_l2_weight_decay(loss_fn, scale=scale)
    return loss_fn
  return loss_ctor


def gaussian_regression_loss(num_index_samples: int,
                             noise_scale: float = 1,
                             l2_weight_decay: float = 0,
                             exclude_bias_l2: bool = True) -> LossCtor:
  """Add a matching Gaussian noise to the target y."""
  def loss_ctor(prior: testbed_base.PriorKnowledge,
                enn: enn_base.EpistemicNetwork) -> enn_base.LossFn:
    """Add a matching Gaussian noise to the target y."""
    noise_std = noise_scale * prior.noise_std
    noise_fn = data_noise.GaussianTargetNoise(enn, noise_std)
    single_loss = losses.add_data_noise(losses.L2Loss(), noise_fn)
    loss_fn = losses.average_single_index_loss(single_loss, num_index_samples)
    if l2_weight_decay != 0:
      if exclude_bias_l2:
        predicate = lambda module, name, value: name != 'b'
      else:
        predicate = lambda module, name, value: True
      loss_fn = losses.add_l2_weight_decay(loss_fn, l2_weight_decay, predicate)
    return loss_fn
  return loss_ctor


def regularized_dropout_loss(num_index_samples: int = 10,
                             dropout_rate: float = 0.05,
                             scale: float = 1e-2,
                             tau: float = 1.0) -> LossCtor:
  """Constructs the special regularized loss of the paper "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning" (2015)."""
  def loss_ctor(prior: testbed_base.PriorKnowledge,
                enn: enn_base.EpistemicNetwork) -> enn_base.LossFn:
    del enn  # Unused
    single_loss = losses.L2Loss()
    reg = (scale ** 2) * (1 - dropout_rate) / (2. * prior.num_train * tau)
    loss_fn = losses.average_single_index_loss(single_loss, num_index_samples)
    return losses.add_l2_weight_decay(loss_fn, scale=reg)
  return loss_ctor


def bbb_loss(sigma_0: float = 100, num_index_samples: int = 64):
  """Constructs the loss function for bbb agent."""
  def loss_ctor(prior: testbed_base.PriorKnowledge,
                enn: enn_base.EpistemicNetwork) -> enn_base.LossFn:
    del enn
    log_likelihood_fn = losses.get_awgn_loglike_fn(prior.noise_std)

    model_prior_kl_fn = losses.get_analytical_diagonal_linear_model_prior_kl_fn(
        prior.num_train, sigma_0)
    single_loss = losses.ElboLoss(log_likelihood_fn, model_prior_kl_fn)
    loss_fn = losses.average_single_index_loss(
        single_loss,
        num_index_samples=num_index_samples)
    return loss_fn
  return loss_ctor
