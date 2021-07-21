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

from typing import Callable

import chex
from enn import base
from enn import utils
from enn.losses import single_index
import haiku as hk
import jax
import jax.numpy as jnp


def get_awgn_loglike_fn(sigma_w: float
                        ) -> Callable[[base.Output, base.Batch], float]:
  """Returns a function that computes the simple unnormalized log likelihood.

  It assumes response variable is perturbed with additive iid Gaussian noise.

  Args:
    sigma_w: standard deviation of the additive Gaussian noise.

  Returns:
    A function that computes the log likelihood given data and output.

  """

  def log_likelihood_fn(out: base.Output, batch: base.Batch):
    chex.assert_shape(batch.y, (None, 1))
    err_sq = jnp.mean(jnp.square(utils.parse_net_output(out) - batch.y))
    return -0.5 * err_sq / sigma_w**2

  return log_likelihood_fn


def get_categorical_loglike_fn(num_classes: int
                               ) -> Callable[[base.Output, base.Batch], float]:
  """Returns a function that computes the unnormalized log likelihood.

  It assumes response variable has a categorical distribution.

  Args:
    num_classes: number of classes for the output.

  Returns:
    A function that computes the log likelihood given data and prediction.

  """

  def log_likelihood_fn(out: base.Output, batch: base.Batch):
    chex.assert_shape(batch.y, (None, 1))
    logits = utils.parse_net_output(out)
    labels = jax.nn.one_hot(batch.y[:, 0], num_classes)
    return jnp.mean(
        jnp.sum(labels * jax.nn.log_softmax(logits), axis=1))

  return log_likelihood_fn


def normal_log_prob(latent: base.Array, sigma: float = 1, mu: float = 0):
  """Compute un-normalized log probability of a normal RV."""
  latent, _ = jax.tree_flatten(latent)
  latent = jax.tree_map(lambda x: x.flatten(), latent)
  latent = jnp.concatenate(latent)
  latent_dim = len(latent)
  latent_l2_sq = jnp.sum(jnp.square(latent - mu))
  return -0.5 * (latent_dim * jnp.log(2 * jnp.pi * sigma**2)
                 + latent_l2_sq / sigma**2)


def get_nn_params_log_prior_prob_fn(
    sigma_0: float) -> Callable[[jnp.array], float]:
  """Returns a function that computes params prior log likelihood from output.

  It assumes that the network output's extra field has an element with the key
  `generated_params` that represents the latent variable.
  It also assumes that index is Gaussian.

  Args:
    sigma_0: standard deviation of the index.
  """

  def log_prob_fn(out: base.Output):
    latent = out.extra['hyper_net_out']
    # weights and biases are assumed to be normalized such that they have the
    # same variance.
    return normal_log_prob(latent, sigma=sigma_0)

  return log_prob_fn


def get_lhm_log_model_prob_fn(
    sigma_z: float) -> Callable[[base.Output, hk.Params, base.Index], float]:
  """Returns a function for log probability of latent under the model.

  It assumes index to be Gaussian with standard deviation sigma_z.
  Args:
    sigma_z: index standard deviation.
  """

  def log_prob_fn(out: base.Output, params: hk.Params, index: base.Index):
    del out  # Here we compute the log prob from params and index directly.
    predicate = lambda module_name, name, value: name == 'w'
    weight_matrices = hk.data_structures.filter(predicate, params)
    weight_matrices, _ = jax.tree_flatten(weight_matrices)
    weight_matrix = jnp.concatenate(weight_matrices, axis=1)
    weight_matrix_sq = jnp.matmul(weight_matrix, weight_matrix.T)
    _, log_det_w_sq = jnp.linalg.slogdet(weight_matrix_sq + 1e-6 *
                                         jnp.eye(weight_matrix_sq.shape[0]))
    log_det_w = 0.5 * log_det_w_sq
    index_l2_sq = jnp.sum(jnp.square(index))
    return -0.5 * index_l2_sq / sigma_z**2 - log_det_w

  return log_prob_fn


def get_diagonal_linear_hypermodel_elbo_fn(
    log_likelihood_fn: Callable[[base.Output, base.Batch], float],
    sigma_0: float,
    num_samples: float) -> single_index.ElboLoss:
  """Returns the negative ELBO for diagonal linear hypermodels.

  Args:
    log_likelihood_fn: log likelihood function.
    sigma_0: Standard deviation of the Gaussian latent (params) prior.
    num_samples: effective number of samples.
  Returns:
    Negative ELBO value.
  """

  def model_prior_kl_fn(
      out: base.Output,
      params: hk.Params,
      index: base.Index) -> float:
    """Compute the KL distance between model and prior densities in a linear HM.

    weights `w` and biases `b` are assumed included in `params`. The latent
    variables (which are the parameters of the base network) are generated as u
    = z @ w + b where z is the index variable. The index is assumed Gaussian
    *with variance equal to the prior variance* of the latent variables.

    This function also  assumes a Gaussian prior distribution for the latent,
    i.e., parameters of the base network.

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
        jnp.sum(jnp.square(scales))
        + jnp.sum(jnp.square(biases)) / (sigma_0 ** 2)
        - len(biases)
        - 2 * jnp.sum(jnp.log(scales))
        )

  return single_index.ElboLoss(
      log_likelihood_fn=log_likelihood_fn,
      model_prior_kl_fn=model_prior_kl_fn)


def get_linear_hypermodel_elbo_fn(
    log_likelihood_fn: Callable[[base.Output, base.Batch], float],
    sigma_0: float,
    num_samples: float) -> single_index.ElboLoss:
  """Returns a loss function that computes the ELBO for linear hypermodels.

  Args:
    log_likelihood_fn: log likelihood function.
    sigma_0: Standard deviation of the Gaussian latent (params) prior.
    num_samples: effective number of samples.
  Returns:
    Negative ELBO value.
  """

  def model_prior_kl_fn(
      out: base.Output,
      params: hk.Params,
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

  return single_index.ElboLoss(
      log_likelihood_fn=log_likelihood_fn,
      model_prior_kl_fn=model_prior_kl_fn)


def get_hyperflow_elbo_fn(
    log_likelihood_fn: Callable[[base.Output, base.Batch], float],
    sigma_0: float,
    num_samples: float) -> single_index.ElboLoss:
  """Returns a loss function that computes the ELBO for hyperflows.

  Args:
    log_likelihood_fn: log likelihood function.
    sigma_0: Standard deviation of the Gaussian latent (params) prior.
    num_samples: effective number of samples.
  Returns:
    loss function computing negative ELBO.
  """

  def model_prior_kl_fn(out, params, index):
    del params, index
    return (jnp.squeeze(out.extra['log_prob'])
            - normal_log_prob(out.extra['latent'], sigma_0)) / num_samples

  loss_fn = single_index.ElboLoss(
      log_likelihood_fn=log_likelihood_fn,
      model_prior_kl_fn=model_prior_kl_fn)

  return loss_fn
