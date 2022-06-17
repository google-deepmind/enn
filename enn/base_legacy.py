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
from typing import Generic, Tuple
import warnings

import chex
from enn import base
import haiku as hk
import typing_extensions

# WARNING: THIS FILE IS DEPRECATED, PLEASE MIGRATE TO base.py
warnings.warn('Legacy interface is deprecated, please move to base.py',
              DeprecationWarning, stacklevel=2)

Array = chex.Array
DataIndex = base.DataIndex  # Always integer elements
Index = base.Index  # Epistemic index, paired with network
RngKey = chex.PRNGKey  # Integer pairs, see jax.random
Input = base.Input

Batch = base.Batch
BatchIterator = base.BatchIterator
LossMetrics = base.LossMetrics
Data = base.Data

OutputWithPrior = base.OutputWithPrior
Output = base.Output


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


class LossFn(typing_extensions.Protocol):
  """Calculates a loss based on one batch of data per rng_key."""

  def __call__(self,
               enn: EpistemicNetwork,
               params: hk.Params,
               batch: Batch,
               key: RngKey) -> LossOutput:
    """Computes a loss based on one batch of data and a random key."""

################################################################################
# Definitions for networks with "state" e.g. BatchNorm.
################################################################################
ApplyFnWithStateBase = base.ApplyFn
InitFnWithStateBase = base.InitFn
EpistemicNetworkWithStateBase = base.EpistemicNetwork
LossFnWithStateBase = base.LossFn
LossOutputWithState = base.LossOutput

# Modules specialized to work only with Array inputs.
ApplyFnWithState = ApplyFnWithStateBase[Array]
InitFnWithState = InitFnWithStateBase[Array]
EpistemicNetworkWithState = EpistemicNetworkWithStateBase[Array]

# LossFnWithStateBase specialized to work only with Array inputs and Batch data.
LossFnWithState = LossFnWithStateBase[Array, Batch]
