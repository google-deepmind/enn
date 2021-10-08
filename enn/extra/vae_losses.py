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
"""Tools for computing VI loss."""
import dataclasses
from typing import Callable, Tuple

import chex
from enn import base
import haiku as hk
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions


# TODO(smaghari): Migrate this stuff to the base/LossFn format
VaeOutput = Tuple[base.Array, base.Array, base.Array, base.Array]
VaeApplyFn = Callable[[hk.Params, base.Array, base.RngKey], VaeOutput]


def binary_log_likelihood(x: base.Array,
                          output: base.Array) -> base.Array:
  """Computes the binary log likelihood loss.

  Computes the reconstruction error for Bernoulli decoder based on Appendix C
  of (Kingma & Welling, 2014).

  Args:
    x: A batch of 1D binary inputs.
    output: A batch of output logits of the network.

  Returns:
    A batch of binary log likelihood losses.
  """
  assert x.ndim == 2
  chex.assert_equal_shape([x, output])

  log_likelihood = jnp.sum(
      x * output - jnp.logaddexp(0.0, output), axis=-1)

  chex.assert_shape(log_likelihood, (x.shape[0],))
  return log_likelihood


def gaussian_log_likelihood(x: base.Array, mean: base.Array,
                            log_variance: base.Array) -> base.Array:
  """Computes the gaussian log likelihood loss.

  Computes the reconstruction error for Gaussian decoder based on Appendix C
  of (Kingma & Welling, 2014).

  Args:
    x: A batch of 1D standardized inputs.
    mean: A batch of mean of the output variable.
    log_variance: A batch of log of the variance of the output variable.

  Returns:
    A batch of gaussian log likelihood losses.
  """
  assert x.ndim == 2
  chex.assert_equal_shape([x, mean, log_variance])

  def log_normal_prob(x: float, mu: float = 0, sigma: float = 1):
    """Compute log probability of x w.r.t a 1D Gaussian density."""
    gauss = tfd.Normal(loc=mu, scale=sigma)
    return gauss.log_prob(x)

  log_normal_prob_vectorized = jnp.vectorize(log_normal_prob)
  log_likelihoods = log_normal_prob_vectorized(x, mean,
                                               jnp.exp(0.5 * log_variance))
  log_likelihood = jnp.sum(log_likelihoods, axis=-1)

  chex.assert_shape(log_likelihood, (x.shape[0],))
  return log_likelihood


def latent_kl_divergence(mean: base.Array,
                         log_variance: base.Array) -> base.Array:
  """Computes the KL divergence of latent distribution w.r.t. Normal(0, I).

  Computes KL divergence of the latent distribution based on Appendix B of
  (Kingma & Welling, 2014).

  Args:
    mean: A batch of mean of the latent variable.
    log_variance: A batch of log of the variance of the latent variable.

  Returns:
    A batch of KL divergences.
  """
  assert mean.ndim == 2
  chex.assert_equal_shape([mean, log_variance])

  kl = 0.5 * jnp.sum(
      1. + log_variance - jnp.square(mean) - jnp.exp(log_variance), axis=-1)
  chex.assert_shape(kl, (mean.shape[0],))
  return kl


@dataclasses.dataclass
class VaeLoss:
  """Evaluates the accuracy error of a greedy logit predictor."""
  bernoulli_decoder: bool

  def __call__(self,
               apply: VaeApplyFn,
               params: hk.Params,
               x: base.Array,
               key: base.RngKey) -> Tuple[base.Array, base.LossMetrics]:
    output_mean, output_log_variance, latent_mean, latent_log_variance = apply(
        params, key, x)
    kl_term = latent_kl_divergence(latent_mean, latent_log_variance)
    if self.bernoulli_decoder:
      log_likelihood = binary_log_likelihood(x, output_mean)
    else:
      log_likelihood = gaussian_log_likelihood(x, output_mean,
                                               output_log_variance)

    chex.assert_equal_shape([kl_term, log_likelihood])
    elbo = kl_term + log_likelihood
    chex.assert_shape(elbo, (x.shape[0],))

    return -jnp.mean(elbo), {}
