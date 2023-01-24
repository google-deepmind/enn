# pylint: disable=g-bad-file-header
# Copyright 2023 DeepMind Technologies Limited. All Rights Reserved.
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
"""Base class for active learning."""

import abc
import typing as tp

import chex
from enn import base as enn_base
from enn import networks
import haiku as hk
import typing_extensions


class ActiveLearner(abc.ABC):
  """Samples a batch from a pool of data for learning.

  An active learner selects an "acquisition batch" with acquisition_size
  elements from a "candidate batch" that is passed to sample_batch. This can be
  used to prioritize data for learning.
  """

  @abc.abstractmethod
  def sample_batch(
      self,
      params: hk.Params,
      state: hk.State,
      batch: enn_base.Batch,
      key: chex.PRNGKey,
  ) -> enn_base.Batch:
    """Samples a batch from a pool of data for learning."""

  @property
  @abc.abstractmethod
  def acquisition_size(self) -> int:
    """Return the acquisition size for the active learner."""

  @acquisition_size.setter
  @abc.abstractmethod
  def acquisition_size(self, size: int) -> None:
    """Overwrites the acquisition size. Useful when we pmap sample_batch."""


PriorityOutput = tp.Tuple[chex.Array, tp.Dict[str, chex.Array]]


class PriorityFn(typing_extensions.Protocol):

  def __call__(
      self,
      params: hk.Params,
      state: hk.State,
      batch: enn_base.Batch,
      key: chex.PRNGKey,
  ) -> PriorityOutput:
    """Assigns a priority score to a batch."""


class PriorityFnCtor(typing_extensions.Protocol):

  def __call__(
      self,
      enn_batch_fwd: networks.EnnBatchFwd[chex.Array],
  ) -> PriorityFn:
    """Constructs a priority function base on an enn_batch_fwd."""


class PerExamplePriority(typing_extensions.Protocol):
  """Interface for priority per example."""

  def __call__(
      self,
      logits: chex.Array,
      labels: chex.Array,
      key: chex.Array,
  ) -> chex.Array:
    """Calculates a priority score based on logits, labels, and a random key.

    Args:
      logits: An array of shape [A, B, C] where B is the batch size of data, C
        is the number of outputs per data (for classification, this is equal to
        number of classes), and A is the number of random samples for each data.
      labels: An array of shape [B, 1] where B is the batch size of data.
      key: A random key.

    Returns:
      A priority score per example of shape [B,].
    """
