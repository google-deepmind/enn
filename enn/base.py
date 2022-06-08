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

"""Base classes for Epistemic Neural Network design in JAX / Haiku."""
# TODO(author2, author3): Slim down the base interfaces.
# Current code has drifted away from desired code quality with feature creep.
# We want to rethink this interface and get it down to something more clear.

import abc
import dataclasses
from typing import Any, Dict, Iterator, NamedTuple, Optional, Tuple, Union, TypeVar, Generic

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import typing_extensions

Array = Union[np.ndarray, jnp.DeviceArray]
DataIndex = Array  # Always integer elements
Index = Any  # Epistemic index, paired with network
RngKey = jnp.DeviceArray  # Integer pairs, see jax.random
# Defining Input as a generic type for the input data. This allows our base
# methods and classes to work with different types of input data, not only Array
# defined here.
Input = TypeVar('Input')


# TODO(author3): Make Batch generic in input. This requires multiple
# inheritance for NamedTuple which is not supported in Python 3.9 yet:
# https://bugs.python.org/issue43923
class Batch(NamedTuple):
  x: Array  # Inputs
  y: Array  # Targets
  data_index: Optional[DataIndex] = None  # Integer identifiers for data
  weights: Optional[Array] = None  # None should default to weights = jnp.ones
  extra: Dict[str, Array] = {}  # You can put other optional stuff here

BatchIterator = Iterator[Batch]  # Equivalent to the dataset we loop through
LossMetrics = Dict[str, Array]
# Defining Data as a generic type for a batch of data. This allows our base
# methods and classes to work with different types of data batch, not only Batch
# defined here.
Data = TypeVar('Data')


class OutputWithPrior(NamedTuple):
  """Output wrapper for networks with prior functions."""
  train: Array
  prior: Array = np.zeros(1)
  extra: Dict[str, Array] = {}

  @property
  def preds(self) -> Array:
    return self.train + jax.lax.stop_gradient(self.prior)

Output = Union[Array, OutputWithPrior]


class EpistemicModule(abc.ABC, hk.Module):
  """Epistemic neural network abstract base class as Haiku module."""

  @abc.abstractmethod
  def __call__(self, inputs: Array, index: Index) -> Output:
    """Forwards the epsitemic network y = f(x,z)."""


################################################################################
# Definitions for networks without "state".
################################################################################
class ApplyFnBase(typing_extensions.Protocol[Input]):
  """Applies the ENN at given parameters, inputs, index."""

  def __call__(self, params: hk.Params, inputs: Input, index: Index) -> Output:
    """Applies the ENN at given parameters, inputs, index."""


class InitFnBase(typing_extensions.Protocol[Input]):
  """Initializes the ENN at given rng_key, inputs, index."""

  def __call__(self, rng_key: RngKey, inputs: Input, index: Index) -> hk.Params:
    """Initializes the ENN at given rng_key, inputs, index."""


class EpistemicIndexer(typing_extensions.Protocol):
  """Generates indices for the ENN from random keys."""

  def __call__(self, key: RngKey) -> Index:
    """Samples a single index for the epistemic network."""


@dataclasses.dataclass
class EpistemicNetworkBase(Generic[Input]):
  """Convenient pairing of Haiku transformed function and index sampler."""
  apply: ApplyFnBase[Input]
  init: InitFnBase[Input]
  indexer: EpistemicIndexer


# Modules specialized to work only with Array inputs.
ApplyFn = ApplyFnBase[Array]
InitFn = InitFnBase[Array]
EpistemicNetwork = EpistemicNetworkBase[Array]

LossOutput = Tuple[Array, LossMetrics]


class LossFnBase(typing_extensions.Protocol[Input, Data]):
  """Calculates a loss based on one batch of data per rng_key."""

  def __call__(self,
               enn: EpistemicNetworkBase[Input],
               params: hk.Params,
               batch: Data,
               key: RngKey) -> LossOutput:
    """Computes a loss based on one batch of data and a random key."""

# LossFnBase specialized to work only with Array inputs and Batch data.
LossFn = LossFnBase[Array, Batch]


################################################################################
# Definitions for networks with "state" e.g. BatchNorm.
################################################################################
class ApplyFnWithStateBase(typing_extensions.Protocol[Input]):
  """Applies the ENN at given parameters, state, inputs, index."""

  def __call__(
      self,
      params: hk.Params,
      state: hk.State,
      inputs: Input,
      index: Index,
  ) -> Tuple[Output, hk.State]:
    """Applies the ENN at given parameters, state, inputs, index."""


class InitFnWithStateBase(typing_extensions.Protocol[Input]):
  """Initializes the ENN with state at given rng_key, inputs, index."""

  def __call__(
      self,
      rng_key: RngKey,
      inputs: Input,
      index: Index,
  ) -> Tuple[hk.Params, hk.State]:
    """Initializes the ENN with state at given rng_key, inputs, index."""


@dataclasses.dataclass
class EpistemicNetworkWithStateBase(Generic[Input]):
  """Convenient pairing of Haiku transformed function and index sampler."""
  apply: ApplyFnWithStateBase[Input]
  init: InitFnWithStateBase[Input]
  indexer: EpistemicIndexer


# Modules specialized to work only with Array inputs.
ApplyFnWithState = ApplyFnWithStateBase[Array]
InitFnWithState = InitFnWithStateBase[Array]
EpistemicNetworkWithState = EpistemicNetworkWithStateBase[Array]

LossOutputWithState = Tuple[Array, Tuple[hk.State, LossMetrics]]


class LossFnWithStateBase(typing_extensions.Protocol[Input, Data]):
  """Calculates a loss based on one batch of data per rng_key."""

  def __call__(self,
               enn: EpistemicNetworkWithStateBase[Input],
               params: hk.Params,
               state: hk.State,
               batch: Data,
               key: RngKey) -> LossOutputWithState:
    """Computes a loss based on one batch of data and a random key."""


# LossFnWithStateBase specialized to work only with Array inputs and Batch data.
LossFnWithState = LossFnWithStateBase[Array, Batch]
