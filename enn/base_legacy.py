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

from typing import Tuple
import warnings

import chex
from enn import base
from enn.losses import base as losses_base
from enn.networks import base as network_base

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

EpistemicModule = network_base.EpistemicModule
EpistemicIndexer = base.EpistemicIndexer

ApplyFn = network_base.ApplyFn
InitFn = network_base.InitFn
EpistemicNetwork = network_base.EpistemicNetwork

LossOutput = Tuple[Array, LossMetrics]
LossFn = losses_base.LossFn


ApplyFnWithStateBase = base.ApplyFn
InitFnWithStateBase = base.InitFn
EpistemicNetworkWithStateBase = base.EpistemicNetwork
LossFnWithStateBase = base.LossFn
LossOutputWithState = base.LossOutput

# Modules specialized to work only with Array inputs.
ApplyFnWithState = network_base.ApplyFnWithState
InitFnWithState = network_base.InitFnWithState
EpistemicNetworkWithState = network_base.EpistemicNetworkWithState

# LossFnWithStateBase specialized to work only with Array inputs and Batch data.
LossFnWithState = LossFnWithStateBase[Array, Batch]
