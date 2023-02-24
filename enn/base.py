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

import dataclasses
import typing as tp

import chex
import haiku as hk
import typing_extensions

################################################################################
# ENN definition

Input = tp.TypeVar('Input')  # Inputs to neural network
Output = tp.TypeVar('Output')  # Outputs of neural network
Index = tp.Any  # Epistemic index


class EpistemicIndexer(typing_extensions.Protocol):
  """Generates indices for the ENN from random keys."""

  def __call__(self, key: chex.PRNGKey) -> Index:
    """Samples a single index for the epistemic network."""


class ApplyFn(typing_extensions.Protocol[Input, Output]):
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
class EpistemicNetwork(tp.Generic[Input, Output]):
  """Convenient pairing of Haiku transformed function and index sampler."""
  apply: ApplyFn[Input, Output]
  init: InitFn[Input]
  indexer: EpistemicIndexer


################################################################################
# Loss functions and training
DataIndex = chex.Array  # Integer identifiers used for bootstrapping
Data = tp.TypeVar('Data')  # General training data
LossMetrics = tp.Dict[str, chex.Array]  # Metrics reported in training.

# Output of loss function includes (loss, (state, metrics))
LossOutput = tp.Tuple[chex.Array, tp.Tuple[hk.State, LossMetrics]]


class LossFn(typing_extensions.Protocol[Input, Output, Data]):
  """Calculates a loss based on one batch of data per random key."""

  def __call__(self,
               enn: EpistemicNetwork[Input, Output],
               params: hk.Params,
               state: hk.State,
               batch: Data,
               key: chex.PRNGKey) -> LossOutput:
    """Computes a loss based on one batch of data and a random key."""


@chex.dataclass(frozen=True)
class Batch:
  x: chex.Array  # Inputs
  y: chex.Array  # Targets
  data_index: tp.Optional[DataIndex] = None  # Integer identifiers for data
  weights: tp.Optional[chex.Array] = None  # None defaults to weights = jnp.ones
  extra: tp.Dict[str, chex.Array] = dataclasses.field(
      default_factory=dict
  )  # You can put other optional stuff here


BatchIterator = tp.Iterator[Batch]  # Equivalent to the dataset we loop through
