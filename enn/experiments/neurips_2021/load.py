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
"""Loading a GP regression instance for the testbed."""
import dataclasses
from typing import Tuple

import chex
from enn.experiments.neurips_2021 import base as testbed_base
from enn.experiments.neurips_2021 import testbed
import haiku as hk
import jax
import jax.config
from neural_tangents import stax
from neural_tangents.utils import typing as nt_types
import numpy as np
import typing_extensions

# TODO(author1): move this config update to an explicit initialize function.
jax.config.update('jax_enable_x64', True)


class KernelCtor(typing_extensions.Protocol):
  """Interface for generating a kernel for a given input dimension."""

  def __call__(self, input_dim: int) -> nt_types.KernelFn:
    """Generates a kernel for a given input dimension."""


@dataclasses.dataclass
class MLPKernelCtor(KernelCtor):
  """Generates a GP kernel corresponding to an infinitely-wide MLP."""
  num_hidden_layers: int
  activation: nt_types.InternalLayer

  def __post_init__(self):
    assert self.num_hidden_layers >= 1, 'Must have at least one hidden layer.'

  def __call__(self, input_dim: int = 1) -> nt_types.KernelFn:
    """Generates a kernel for a given input dimension."""
    limit_width = 50  # Implementation detail of neural_testbed, unused.
    layers = [
        stax.Dense(limit_width, W_std=1, b_std=1 / np.sqrt(input_dim))
    ]
    for _ in range(self.num_hidden_layers - 1):
      layers.append(self.activation)
      layers.append(stax.Dense(limit_width, W_std=1, b_std=0))
    layers.append(self.activation)
    layers.append(stax.Dense(1, W_std=1, b_std=0))
    _, _, kernel = stax.serial(*layers)
    return kernel


def make_benchmark_kernel(input_dim: int = 1) -> nt_types.KernelFn:
  """Creates the benchmark kernel used in the testbed = 2-layer ReLU."""
  kernel_ctor = MLPKernelCtor(num_hidden_layers=2, activation=stax.Relu())
  return kernel_ctor(input_dim)


def gaussian_data(key: chex.PRNGKey,
                  num_train: int,
                  input_dim: int,
                  num_test: int) -> Tuple[chex.Array, chex.Array]:
  """Generate Gaussian training and test data."""
  train_key, test_key = jax.random.split(key)
  x_train = jax.random.normal(train_key, [num_train, input_dim])
  x_test = jax.random.normal(test_key, [num_test, input_dim])
  return x_train, x_test


@dataclasses.dataclass
class RegressionTestbedConfig:
  """Configuration options for regression testbed instance."""
  num_train: int
  input_dim: int
  seed: int
  noise_std: float
  tau: int = 1  # TODO(author2): Consider design of this parameter.
  num_test_cache: int = 1000
  target_test_seeds: int = 1000  # num_test_seeds = target_test_seeds / tau
  num_enn_samples: int = 100
  kernel_ctor: KernelCtor = make_benchmark_kernel
  num_layers: int = 1  # Output to prior knowledge


def regression_load_from_config(
    config: RegressionTestbedConfig) -> testbed_base.TestbedProblem:
  """Loads regression problem from config."""
  rng = hk.PRNGSequence(config.seed)
  x_train, x_test = gaussian_data(
      key=next(rng),
      num_train=config.num_train,
      input_dim=config.input_dim,
      num_test=config.num_test_cache,
  )
  data_sampler = testbed.GPRegression(
      kernel_fn=config.kernel_ctor(config.input_dim),
      x_train=x_train,
      x_test=x_test,
      tau=config.tau,
      noise_std=config.noise_std,
      key=next(rng),
  )
  prior_knowledge = testbed_base.PriorKnowledge(
      input_dim=config.input_dim,
      num_train=config.num_train,
      num_classes=1,
      tau=1,
      layers=config.num_layers,
      noise_std=config.noise_std,
  )
  assert config.tau == 1, 'Only works for tau=1'
  return testbed.TestbedGPRegression(
      data_sampler,
      prior_knowledge,
      key=next(rng),
      num_enn_samples=config.num_enn_samples)


def regression_load(input_dim: int,
                    data_ratio: float,
                    seed: int,
                    noise_std: float) -> testbed_base.TestbedProblem:
  """Load GP regression from sweep hyperparameters."""
  num_train = int(data_ratio * input_dim)
  config = RegressionTestbedConfig(num_train, input_dim, seed, noise_std)
  return regression_load_from_config(config)
