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

from typing import Callable

import chex
from enn import base
from enn import networks
import haiku as hk
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

import typing_extensions
tfd = tfp.distributions


def get_awgn_loglike_fn(
    sigma_w: float) -> Callable[[networks.Output, base.Batch], float]:
  """Returns a function that computes the simple unnormalized log likelihood.

  It assumes response variable is perturbed with additive iid Gaussian noise.

  Args:
    sigma_w: standard deviation of the additive Gaussian noise.

  Returns:
    A function that computes the log likelihood given data and output.
  """

  def log_likelihood_fn(out: networks.Output, batch: base.Batch):
    chex.assert_shape(batch.y, (None, 1))
    err_sq = jnp.mean(jnp.square(networks.parse_net_output(out) - batch.y))
    return -0.5 * err_sq / sigma_w**2

  return log_likelihood_fn


def get_categorical_loglike_fn(
    num_classes: int
) -> Callable[[networks.Output, base.Batch], float]:
  """Returns a function that computes the unnormalized log likelihood.

  It assumes response variable has a categorical distribution.

  Args:
    num_classes: number of classes for the output.

  Returns:
    A function that computes the log likelihood given data and prediction.
  """

  def log_likelihood_fn(out: networks.Output, batch: base.Batch):
    chex.assert_shape(batch.y, (None, 1))
    logits = networks.parse_net_output(out)
    labels = jax.nn.one_hot(batch.y[:, 0], num_classes)
    return jnp.mean(
        jnp.sum(labels * jax.nn.log_softmax(logits), axis=1))

  return log_likelihood_fn


def log_normal_prob(x: float, mu: float = 0, sigma: float = 1):
  """Compute log probability of x w.r.t a 1D Gaussian density."""
  gauss = tfd.Normal(loc=mu, scale=sigma)
  return gauss.log_prob(x)


def sum_log_scale_mixture_normal(
    x: chex.Array,
    sigma_1: float,
    sigma_2: float,
    mu_1: float = 0.,
    mu_2: float = 0.,
    pi: float = 1.,
) -> float:
  """Compute sum log probs of x w.r.t. a scale mixture of two 1D Gaussians.

  Args:
    x: an array for which we want to find probabilities.
    sigma_1: Standard deviation of the first Gaussian denisty.
    sigma_2: Standard deviation of the second Gaussian.
    mu_1: Mean of the first Gaussian denisty.
    mu_2: Mean of the second Gaussian denisty.
    pi: Scale for mixture of two Gaussian densities. The two Gaussian
      densities are mixed as
      pi * Normal(mu_1, sigma_1) + (1 - pi) * Normal(mu_2, sigma_2)
  Returns:
    Sum of log probabilities.
  """
  bimix_gauss = tfd.MixtureSameFamily(
      mixture_distribution=tfd.Categorical(probs=[pi, 1.-pi]),
      components_distribution=tfd.Normal(
          loc=[mu_1, mu_2],  # One for each component.
          scale=[sigma_1, sigma_2]))
  log_probs = bimix_gauss.log_prob(x)
  return jnp.sum(log_probs)


def normal_log_prob(latent: chex.Array, sigma: float = 1, mu: float = 0):
  """Compute un-normalized log probability of a normal RV."""
  latent, _ = jax.tree_flatten(latent)
  latent = jax.tree_map(lambda x: x.flatten(), latent)
  latent = jnp.concatenate(latent)
  latent_dim = len(latent)
  latent_l2_sq = jnp.sum(jnp.square(latent - mu))
  return -0.5 * (latent_dim * jnp.log(2 * jnp.pi * sigma**2)
                 + latent_l2_sq / sigma**2)


class _KlLossFn(typing_extensions.Protocol):
  """Calculates a loss based on model output, params, and one index.."""

  def __call__(
      self,
      out: networks.Output,
      params: hk.Params,
      index: base.Index,
  ) -> float:
    """Computes a loss based on model output, params, and one index."""


def get_sample_based_model_prior_kl_fn(
    num_samples: float, sigma_1: float, sigma_2: float = 1., scale: float = 1.
) -> _KlLossFn:
  """Returns a function for computing the KL distance between model and prior.

  Args:
    num_samples: effective number of samples.
    sigma_1: Standard deviation of the Gaussian denisty as the prior.
    sigma_2: Standard deviation of the second Gaussian if scale mixture of two
      Gaussian densities is used as the prior.
    scale: Scale for mixture of two Gaussian densities. The two Gaussian
      densities are mixed as
      scale * Normal(0, sigma_1) + (1 - scale) * Normal(0, sigma_2)
  """

  def model_prior_kl_fn(out: networks.Output, params: hk.Params,
                        index: base.Index) -> float:
    """Compute the KL distance between model and prior densities using samples."""
    del index
    latent = out.extra['hyper_index']

    # Calculate prior
    log_priors_sum = sum_log_scale_mixture_normal(latent, sigma_1,
                                                  sigma_2, pi=scale)

    # Calculate variational posterior
    predicate = lambda module_name, name, value: name == 'w'
    # We have used 'w' in params as rho (used with softplus to calculate sigma)
    # and 'b' in params as mu for the Gaussian density.
    rhos, mus = hk.data_structures.partition(predicate, params)
    mus, _ = jax.tree_flatten(mus)
    mus = jnp.concatenate(mus, axis=0)
    rhos, _ = jax.tree_flatten(rhos)
    rhos = jnp.concatenate(rhos, axis=0)
    chex.assert_equal_shape([rhos, mus])
    # We use softplus to convert rho to sigma.
    sigmas = jnp.log(1 + jnp.exp(rhos))
    chex.assert_equal_shape([sigmas, mus, latent])
    log_normal_prob_vectorized = jnp.vectorize(log_normal_prob)
    log_var_posteriors = log_normal_prob_vectorized(latent, mus, sigmas)
    log_var_posteriors_sum = jnp.sum(log_var_posteriors)

    return (log_var_posteriors_sum - log_priors_sum) / num_samples

  return model_prior_kl_fn


def get_analytical_diagonal_linear_model_prior_kl_fn(
    num_samples: float, sigma_0: float
) -> _KlLossFn:
  """Returns a function for computing the KL distance between model and prior.

  It assumes index to be standard Gaussian.

  Args:
    num_samples: effective number of samples.
    sigma_0: Standard deviation of the Gaussian latent (params) prior.
  Returns:
    model_prior_kl_fn
  """

  def model_prior_kl_fn(out: networks.Output, params: hk.Params,
                        index: base.Index) -> float:
    """Compute the KL distance between model and prior densities in a linear HM.

    weights `w` and biases `b` are assumed included in `params`. The latent
    variables (which are the parameters of the base network) are generated as u
    = z @ log(1 + exp(w)) + b where z is the index variable. The index is
    assumed to be a standard Gaussian.

    This function also  assumes a Gaussian prior distribution for the latent,
    i.e., parameters of the base network, with variance sigma^2.

    Args:
      out: final output of the hypermodel, i.e., y = f_theta(x, z)
      params: parameters of the hypermodel (Note that this is the parameters of
        the hyper network since base network params are set by the hyper net.)
      index: index z

    Returns:
      KL distance.
    """
    del out, index  # Here we compute the log prob from params directly.
    predicate = lambda module_name, name, value: name == 'w'
    weights, biases = hk.data_structures.partition(predicate, params)
    biases, _ = jax.tree_flatten(biases)
    biases = jnp.concatenate(biases, axis=0)
    weights, _ = jax.tree_flatten(weights)
    weights = jnp.concatenate(weights, axis=0)
    scales = jnp.log(1 + jnp.exp(weights))
    chex.assert_equal_shape([scales, biases])
    return 0.5  / num_samples * (
        jnp.sum(jnp.square(scales)) / (sigma_0 ** 2)
        + jnp.sum(jnp.square(biases)) / (sigma_0 ** 2)
        - len(biases)
        - 2 * jnp.sum(jnp.log(scales))
        + 2 * len(biases) * jnp.log(sigma_0)
        )

  return model_prior_kl_fn


def get_analytical_linear_model_prior_kl_fn(
    num_samples: float, sigma_0: float
) -> _KlLossFn:
  """Returns a function for computing the KL distance between model and prior.

  It assumes index to be Gaussian with standard deviation sigma_0.

  Args:
    num_samples: effective number of samples.
    sigma_0: Standard deviation of the Gaussian latent (params) prior.
  """
  def model_prior_kl_fn(out: networks.Output, params: hk.Params,
                        index: base.Index) -> float:
    """Compute the KL distance between model and prior densities in a linear HM.

    weights `w` and biases `b` are assumed included in `params`. The latent
    variables (which are the parameters of the base network) are generated as u
    = z @ w + b where z is the index variable. The index is assumed Gaussian
    *with variance equal to the prior variance* of the latent variables.

    This function also  assumes a Gaussian prior distribution for the latent,
    i.e., parameters of the base network, and assumes the index to be Gaussian
    *with variance equal to the prior variance* of the latent variables.

    Args:
      out: final output of the hypermodel, i.e., y = f_theta(x, z)
      params: parameters of the hypermodel (Note that this is the parameters of
        the hyper network since base network params are set by the hyper net.)
      index: index z

    Returns:
      KL distance.
    """
    del out, index  # Here we compute the log prob from params directly.
    predicate = lambda module_name, name, value: name == 'w'
    weights, biases = hk.data_structures.partition(predicate, params)
    biases, _ = jax.tree_flatten(biases)
    biases = jnp.concatenate(biases, axis=0)
    weights, _ = jax.tree_flatten(weights)
    weights = jnp.concatenate(weights, axis=1)
    chex.assert_equal_shape_suffix([weights, biases], 1)
    weights_sq = weights @ weights.T
    index_dim = weights_sq.shape[0]
    # Make weights_sq PD for numerical stability
    weights_sq += 1e-6 * jnp.eye(index_dim)

    w_sq_eigvals = jnp.linalg.eigvalsh(weights_sq)
    w_sq_inv = jnp.linalg.inv(weights_sq)
    # Latent covariance is equal to \Sigma_W^2
    sigma_u_log_det = jnp.sum(jnp.log(w_sq_eigvals))
    sigma_u_trace = jnp.sum(w_sq_eigvals)

    weights_biases = weights @ biases
    chex.assert_equal(len(weights_biases), index_dim)
    proj_biases_norm = weights_biases @ w_sq_inv @ weights_biases.T
    return 0.5  / num_samples * (sigma_u_trace - index_dim - sigma_u_log_det
                                 + proj_biases_norm / sigma_0**2)

  return model_prior_kl_fn


def get_analytical_hyperflow_model_prior_kl_fn(
    num_samples: float, sigma_0: float
) -> _KlLossFn:
  """Returns a function for computing the KL distance between model and prior.

  It assumes index to be Gaussian with standard deviation sigma_0.

  Args:
    num_samples: effective number of samples.
    sigma_0: Standard deviation of the Gaussian latent (params) prior.
  """
  def model_prior_kl_fn(out, params, index):
    del params, index
    return (jnp.squeeze(out.extra['log_prob'])
            - normal_log_prob(out.extra['latent'], sigma_0)) / num_samples

  return model_prior_kl_fn
