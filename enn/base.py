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

"""Base classes for Epistemic Neural Network design in JAX / Haiku."""

import abc
import dataclasses
from typing import Any, Dict, Iterator, NamedTuple, Optional, Tuple, Union

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import typing_extensions

Array = Union[np.ndarray, jnp.DeviceArray]
DataIndex = Array  # Always integer elements
Index = Any  # Epistemic index, paired with network
RngKey = jnp.DeviceArray  # Integer pairs, see jax.random


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


class ApplyFn(typing_extensions.Protocol):
  """Applies the ENN at given parameters, inputs, index."""

  def __call__(self, params: hk.Params, inputs: Array, index: Index) -> Output:
    """Applies the ENN at given parameters, inputs, index."""


class InitFn(typing_extensions.Protocol):
  """Initializes the ENN at given rng_key, inputs, index."""

  def __call__(self, rng_key: RngKey, inputs: Array, index: Index) -> hk.Params:
    """Initializes the ENN at given rng_key, inputs, index."""


class EpistemicIndexer(typing_extensions.Protocol):
  """Generates indices for the ENN from random keys."""

  def __call__(self, key: RngKey) -> Index:
    """Samples a single index for the epistemic network."""


@dataclasses.dataclass
class EpistemicNetwork:
  """Convenient pairing of Haiku transformed function and index sampler."""
  apply: ApplyFn
  init: InitFn
  indexer: EpistemicIndexer


class Batch(NamedTuple):
  x: Array  # Inputs
  y: Array  # Targets
  data_index: Optional[DataIndex] = None  # Integer identifiers for data
  weights: Optional[Array] = None  # None should default to weights = jnp.ones
  extra: Dict[str, Array] = {}  # You can put other optional stuff here

BatchIterator = Iterator[Batch]  # Equivalent to the dataset we loop through
LossMetrics = Dict[str, Array]


class LossFn(typing_extensions.Protocol):
  """Calculates a loss based on one batch of data per rng_key."""

  def __call__(self,
               enn: EpistemicNetwork,
               params: hk.Params,
               batch: Batch,
               key: RngKey) -> Tuple[Array, LossMetrics]:
    """Computes a loss based on one batch of data and a random key."""
