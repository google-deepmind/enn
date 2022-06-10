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
"""Base classes for data noise process."""

from enn import base_legacy
import typing_extensions


class DataNoiseBase(typing_extensions.Protocol[base_legacy.Data]):

  def __call__(
      self,
      data: base_legacy.Data,
      index: base_legacy.Index,
  ) -> base_legacy.Data:
    """Apply some noise process to a batch of data based on epistemic index."""


# DataNoiseBase specialized to work only with Batch data.
DataNoise = DataNoiseBase[base_legacy.Batch]


def get_indexer(indexer: base_legacy.EpistemicIndexer):
  while hasattr(indexer, 'indexer'):
    indexer = indexer.indexer
  return indexer
