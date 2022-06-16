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

import dataclasses
import typing as tp

import chex
import haiku as hk
import jax
import numpy as np
import typing_extensions

################################################################################
# ENN definition

Input = tp.TypeVar('Input')  # Inputs to neural network
Index = tp.Any  # Epistemic index


# TODO(author3): Change Output interface to be only OutputWithPrior.
class OutputWithPrior(tp.NamedTuple):
  """Output wrapper for networks with prior functions."""
  train: chex.Array
  prior: chex.Array = np.zeros(1)
  extra: tp.Dict[str, chex.Array] = {}

  @property
  def preds(self) -> chex.Array:
    return self.train + jax.lax.stop_gradient(self.prior)

Output = tp.Union[chex.Array, OutputWithPrior]


class EpistemicIndexer(typing_extensions.Protocol):
  """Generates indices for the ENN from random keys."""

  def __call__(self, key: chex.PRNGKey) -> Index:
    """Samples a single index for the epistemic network."""


class ApplyFn(typing_extensions.Protocol[Input]):
  """Applies the ENN at given parameters, state, inputs, index."""

  def __call__(self,
               params: hk.Params,
               state: hk.State,
               inputs: Input,
               index: Index) -> tp.Tuple[Output, hk.State]:
    """Applies the ENN at given parameters, state, inputs, index."""


class InitFn(typing_extensions.Protocol[Input]):
  """Initializes the ENN with state at given rng_key, inputs, index."""

  def __call__(self,
               rng_key: chex.PRNGKey,
               inputs: Input,
               index: Index) -> tp.Tuple[hk.Params, hk.State]:
    """Initializes the ENN with state at given rng_key, inputs, index."""


@dataclasses.dataclass
class EpistemicNetwork(tp.Generic[Input]):
  """Convenient pairing of Haiku transformed function and index sampler."""
  apply: ApplyFn[Input]
  init: InitFn[Input]
  indexer: EpistemicIndexer


################################################################################
# Loss functions and training
DataIndex = chex.Array  # Integer identifiers used for bootstrapping
Data = tp.TypeVar('Data')  # General training data


# TODO(author3): Make Batch generic in input. This requires multiple
# inheritance for NamedTuple which is not supported in Python 3.9 yet:
# https://bugs.python.org/issue43923
class Batch(tp.NamedTuple):
  x: chex.Array  # Inputs
  y: chex.Array  # Targets
  data_index: tp.Optional[DataIndex] = None  # Integer identifiers for data
  weights: tp.Optional[chex.Array] = None  # None defaults to weights = jnp.ones
  extra: tp.Dict[str, chex.Array] = {}  # You can put other optional stuff here

BatchIterator = tp.Iterator[Batch]  # Equivalent to the dataset we loop through
LossMetrics = tp.Dict[str, chex.Array]  # Metrics reported in training.

# Output of loss function includes (loss, (state, metrics))
LossOutput = tp.Tuple[chex.Array, tp.Tuple[hk.State, LossMetrics]]


class LossFn(typing_extensions.Protocol[Input, Data]):
  """Calculates a loss based on one batch of data per rng_key."""

  def __call__(self,
               enn: EpistemicNetwork[Input],
               params: hk.Params,
               state: hk.State,
               batch: Data,
               key: chex.PRNGKey) -> LossOutput:
    """Computes a loss based on one batch of data and a random key."""

