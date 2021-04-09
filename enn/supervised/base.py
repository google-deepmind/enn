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

"""Base classes for a 'standard' supervised experiment."""
import abc

import dataclasses
from enn import base
import haiku as hk


@dataclasses.dataclass
class BaseExperiment(abc.ABC):
  """Base interface for experiment classes."""
  dataset: base.BatchIterator
  params: hk.Params
  step: int

  @abc.abstractmethod
  def train(self, num_batches: int):
    """Train the ENN for num_batches."""

  @abc.abstractmethod
  def predict(
      self, inputs: base.Array, seed: int) -> base.Array:
    """Evaluate the trained model at given inputs."""

  @abc.abstractmethod
  def loss(self, batch: base.Batch, seed: int) -> base.Array:
    """Calculate the loss at a given batch."""
