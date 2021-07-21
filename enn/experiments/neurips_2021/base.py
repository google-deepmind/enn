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
"""Base classes for GP testbed."""
import abc
from typing import Any, Dict, NamedTuple, Optional

import chex
import dataclasses
import typing_extensions


# Maybe this Data class needs to be a tf.Dataset
class Data(NamedTuple):
  x: chex.Array
  y: chex.Array


@dataclasses.dataclass
class PriorKnowledge:
  input_dim: int
  num_train: int
  num_classes: int = 1
  layers: Optional[int] = None
  noise_std: Optional[float] = None
  temperature: Optional[float] = None
  extra: Optional[Dict[str, Any]] = None


@dataclasses.dataclass
class ENNQuality:
  kl_estimate: float
  extra: Optional[Dict[str, Any]] = None


class EpistemicSampler(typing_extensions.Protocol):
  """Interface for drawing posterior samples from distribution.

  We are considering a model of data: y_i = f(x_i) + e_i.
  In this case the sampler should only model f(x), not aleatoric y.
  """

  def __call__(self, x: chex.Array, seed: int = 0) -> chex.Array:
    """Generate a random sample for epistemic f(x)."""


class TestbedAgent(typing_extensions.Protocol):
  """An interface for specifying a testbed agent."""

  def __call__(self,
               data: Data,
               prior: Optional[PriorKnowledge] = None) -> EpistemicSampler:
    """Sets up a training procedure given ENN prior knowledge."""


class TestbedProblem(abc.ABC):
  """An interface for specifying a generative GP model of data."""

  @abc.abstractproperty
  def train_data(self) -> Data:
    """Access training data from the GP for ENN training."""

  @abc.abstractmethod
  def evaluate_quality(self, enn_sampler: EpistemicSampler) -> ENNQuality:
    """Evaluate the quality of a posterior sampler."""

  @abc.abstractproperty
  def prior_knowledge(self) -> PriorKnowledge:
    """Information describing the problem instance."""

