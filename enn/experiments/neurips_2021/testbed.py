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
"""GP regression testbed problem.

Uses the neural_tangent library to compute the posterior mean and covariance
for regression problem in closed form.
"""
import dataclasses
import chex
from enn.experiments.neurips_2021 import base as testbed_base
import haiku as hk
import jax
import jax.numpy as jnp
import neural_tangents as nt
from neural_tangents._src.utils import typing as nt_types


class GPRegression:
  """GP with gaussian noise output."""

  def __init__(self,
               kernel_fn: nt_types.AnalyticKernelFn,
               x_train: chex.Array,
               x_test: chex.Array,
               key: chex.PRNGKey,
               tau: int = 1,
               noise_std: float = 1,
               kernel_ridge: float = 1e-6):

    # Checking the dimensionality of our data coming in.
    num_train, input_dim = x_train.shape
    num_test_x_cache, input_dim_test = x_test.shape
    assert input_dim == input_dim_test

    rng = hk.PRNGSequence(key)
    self._tau = tau
    self._input_dim = input_dim
    self._x_train = jnp.array(x_train)
    self._x_test = jnp.array(x_test)
    self._num_train = num_train
    self._num_test_x_cache = num_test_x_cache
    self._noise_std = noise_std
    self._kernel_ridge = kernel_ridge

    # Form the training data
    mean = jnp.zeros(num_train)
    k_train_train = kernel_fn(self._x_train, x2=None, get='nngp')
    k_train_train += kernel_ridge * jnp.eye(num_train)
    y_function = jax.random.multivariate_normal(next(rng), mean, k_train_train)
    y_noise = jax.random.normal(next(rng), [num_train, 1]) * noise_std
    y_train = y_function[:, None] + y_noise
    self._train_data = testbed_base.Data(x_train, y_train)
    chex.assert_shape(y_train, [num_train, 1])

    # Form the posterior prediction at cached test data
    predict_fn = nt.predict.gradient_descent_mse_ensemble(
        kernel_fn, x_train, y_train, diag_reg=(noise_std**2))
    self._test_mean, self._test_cov = predict_fn(
        t=None, x_test=self._x_test, get='nngp', compute_cov=True)
    self._test_cov += kernel_ridge * jnp.eye(num_test_x_cache)
    chex.assert_shape(self._test_mean, [num_test_x_cache, 1])
    chex.assert_shape(self._test_cov, [num_test_x_cache, num_test_x_cache])

  @property
  def x_test(self) -> chex.Array:
    return self._x_test

  @property
  def test_mean(self) -> chex.Array:
    return self._test_mean

  @property
  def test_cov(self) -> chex.Array:
    return self._test_cov

  @property
  def train_data(self) -> testbed_base.Data:
    return self._train_data


@dataclasses.dataclass
class TestbedGPRegression(testbed_base.TestbedProblem):
  """Wraps GPRegression sampler for testbed with exact posterior inference."""
  data_sampler: GPRegression
  prior: testbed_base.PriorKnowledge
  key: chex.PRNGKey
  num_enn_samples: int = 100
  std_ridge: float = 1e-3

  @property
  def train_data(self) -> testbed_base.Data:
    return self.data_sampler.train_data

  @property
  def prior_knowledge(self) -> testbed_base.PriorKnowledge:
    return self.prior

  def evaluate_quality(
      self,
      enn_sampler: testbed_base.EpistemicSampler) -> testbed_base.ENNQuality:
    """Computes KL estimate on mean functions for tau=1 only."""
    # Extract useful quantities from the gp sampler.
    x_test = self.data_sampler.x_test
    num_test = x_test.shape[0]
    posterior_mean = self.data_sampler.test_mean[:, 0]
    posterior_std = jnp.sqrt(jnp.diag(self.data_sampler.test_cov))
    posterior_std += self.std_ridge

    # Compute the mean and std of ENN posterior
    batched_sampler = jax.jit(jax.vmap(enn_sampler, in_axes=[None, 0]))
    enn_keys = jax.random.split(self.key, self.num_enn_samples)
    enn_samples = batched_sampler(x_test, enn_keys)
    enn_samples = enn_samples[:, :, 0]
    chex.assert_shape(enn_samples, [self.num_enn_samples, num_test])
    enn_mean = jnp.mean(enn_samples, axis=0)
    enn_std = jnp.std(enn_samples, axis=0) + self.std_ridge

    # Compute the KL divergence between this and reference posterior
    batched_kl = jax.jit(jax.vmap(_kl_gaussian))
    kl_estimates = batched_kl(posterior_mean, posterior_std, enn_mean, enn_std)
    chex.assert_shape(kl_estimates, [num_test])
    kl_estimate = jnp.mean(kl_estimates)
    return testbed_base.ENNQuality(kl_estimate)


def _kl_gaussian(
    mean_1: float, std_1: float, mean_2: float, std_2: float) -> float:
  """Computes the KL(P_1 || P_2) for P_1,P_2 univariate Gaussian."""
  log_term = jnp.log(std_2 / std_1)
  frac_term = (std_1 ** 2 + (mean_1 - mean_2) ** 2) / (2 * std_2 ** 2)
  return log_term + frac_term - 0.5
