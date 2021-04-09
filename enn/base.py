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
from typing import Callable, Dict, Iterator, NamedTuple, Union, Tuple

import dataclasses
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import typing_extensions

Array = Union[np.ndarray, jnp.DeviceArray]
DataIndex = Array  # Always integer
Index = Array  # Epistemic index, paired with network
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

ApplyFn = Callable[[hk.Params, Array, Index], Output]  # Forward params on (x,z)
InitFn = Callable[[RngKey, Array, Index], hk.Params]  # Initialize module params


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


Batch = Union[Dict[str, np.ndarray], NamedTuple]  # Batch of data to train on
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
