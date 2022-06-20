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

"""Base classes for networks."""
import abc
import dataclasses

import chex
from enn import base
import haiku as hk
import typing_extensions


################################################################################
# Sepcializing the network definitions from base.py to work with Array inputs
################################################################################
ApplyArray = base.ApplyFn[chex.Array]
InitArray = base.InitFn[chex.Array]
EnnArray = base.EpistemicNetwork[chex.Array]


# The default network definitions above assume that the network has a state.
# Since a network might not have a state, below we provide definitions for
# Epistemic Networks without state, specialized to work with Array inputs.


class ApplyNoState(typing_extensions.Protocol):
  """Applies the ENN at given parameters, inputs, index."""

  def __call__(self, params: hk.Params,
               inputs: chex.Array,
               index: base.Index) -> base.Output:
    """Applies the ENN at given parameters, inputs, index."""


class InitNoState(typing_extensions.Protocol):
  """Initializes the ENN at given rng_key, inputs, index."""

  def __call__(self, rng_key: chex.PRNGKey,
               inputs: chex.Array,
               index: base.Index) -> hk.Params:
    """Initializes the ENN at given rng_key, inputs, index."""


@dataclasses.dataclass
class EnnNoState:
  """Convenient pairing of Haiku transformed function and index sampler."""
  apply: ApplyNoState
  init: InitNoState
  indexer: base.EpistemicIndexer


class EpistemicModule(abc.ABC, hk.Module):
  """Epistemic neural network abstract base class as Haiku module."""

  @abc.abstractmethod
  def __call__(self, inputs: chex.Array, index: base.Index) -> base.Output:
    """Forwards the epsitemic network y = f(x,z)."""
