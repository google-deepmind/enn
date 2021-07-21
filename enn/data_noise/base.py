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
"""Base classes for data noise process."""

from enn import base
import typing_extensions


class DataNoise(typing_extensions.Protocol):

  def __call__(self, data: base.Batch, index: base.Index) -> base.Batch:
    """Apply some noise process to a batch of data based on epistemic index."""


def get_indexer(indexer: base.EpistemicIndexer):
  while hasattr(indexer, 'indexer'):
    indexer = indexer.indexer
  return indexer
