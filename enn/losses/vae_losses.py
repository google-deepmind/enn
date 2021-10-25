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
"""Tools for computing VAE loss."""
from typing import Callable

import chex
from enn import base
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions


def binary_log_likelihood(x: base.Array, output: base.Array) -> float:
  """Computes the binary log likelihood loss.

  Computes the reconstruction error for Bernoulli decoder based on Appendix C
  of (Kingma & Welling, 2014).

  Args:
    x: A batch of 1D binary inputs.
    output: A batch of output logits (for class 1) of the network.

  Returns:
    Binary log likelihood loss.
  """
  assert x.ndim == 2
  chex.assert_equal_shape([x, output])

  log_likelihood = jnp.sum(
      x * output - jnp.logaddexp(0.0, output), axis=-1)

  chex.assert_shape(log_likelihood, (x.shape[0],))
  return jnp.mean(log_likelihood)


def gaussian_log_likelihood(x: base.Array, mean: base.Array,
                            log_variance: base.Array) -> float:
  """Computes the gaussian log likelihood loss.

  Computes the reconstruction error for Gaussian decoder based on Appendix C
  of (Kingma & Welling, 2014).

  Args:
    x: A batch of 1D standardized inputs.
    mean: A batch of mean of the output variable.
    log_variance: A batch of log of the variance of the output variable.

  Returns:
    Gaussian log likelihood loss.
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
  return jnp.mean(log_likelihood)


def latent_kl_divergence(mean: base.Array,
                         log_variance: base.Array) -> float:
  """Computes the KL divergence of latent distribution w.r.t. Normal(0, I).

  Computes KL divergence of the latent distribution based on Appendix B of
  (Kingma & Welling, 2014).

  Args:
    mean: A batch of mean of the latent variable.
    log_variance: A batch of log of the variance of the latent variable.

  Returns:
    KL divergence.
  """
  assert mean.ndim == 2
  chex.assert_equal_shape([mean, log_variance])

  kl = - 0.5 * jnp.sum(
      1. + log_variance - jnp.square(mean) - jnp.exp(log_variance), axis=-1)
  chex.assert_shape(kl, (mean.shape[0],))
  return jnp.mean(kl)


def get_log_likelihood_fn(
    bernoulli_decoder: bool
) -> Callable[[base.OutputWithPrior, base.Batch], float]:
  """Returns a function for calculating KL divergence of latent distribution.

  Args:
    bernoulli_decoder: A boolean specifying whether the decoder is Bernoulli.
        If it is False, the the decoder is considered to be Gaussian.
  Returns:
    log_likelihood_fn
  """

  def log_likelihood_fn(net_out: base.OutputWithPrior,
                        batch: base.Batch) -> float:
    extra = net_out.extra
    assert 'out_mean' in extra
    assert 'out_log_variance' in extra
    if bernoulli_decoder:
      return binary_log_likelihood(batch.x, extra['out_mean'])
    else:
      return gaussian_log_likelihood(batch.x, extra['out_mean'],
                                     extra['out_log_variance'])

  return log_likelihood_fn


def get_latent_kl_fn() -> Callable[[base.OutputWithPrior], float]:
  """Returns a function for calculating KL divergence of latent distribution."""
  def latent_kl_fn(net_out: base.OutputWithPrior) -> float:
    extra = net_out.extra
    assert 'latent_mean' in extra
    assert 'latent_log_variance' in extra
    return latent_kl_divergence(extra['latent_mean'],
                                extra['latent_log_variance'])

  return latent_kl_fn
