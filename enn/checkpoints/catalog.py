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
"""Entries for pretrained ENNs are stored here."""

import enum

from enn.checkpoints import base as checkpoint_base
from enn.checkpoints import epinet as checkpoint_epinet
from enn.checkpoints import imagenet


# Alias to fit inside character limit
_EnnCpt = checkpoint_base.EnnCheckpoint
_EpiCpt = checkpoint_epinet.EpinetCheckpoint


class ImagenetModels(enum.Enum):
  """Pretrained models on ImageNet."""
  RESNET_50: _EnnCpt = imagenet.resnet_50()
  RESNET_101: _EnnCpt = imagenet.resnet_101()
  RESNET_152: _EnnCpt = imagenet.resnet_152()
  RESNET_200: _EnnCpt = imagenet.resnet_200()
  RESNET_50_FINAL_EPINET: _EpiCpt = imagenet.resnet_50_final_epinet()
  RESNET_101_FINAL_EPINET: _EpiCpt = imagenet.resnet_101_final_epinet()
  RESNET_152_FINAL_EPINET: _EpiCpt = imagenet.resnet_152_final_epinet()
  RESNET_200_FINAL_EPINET: _EpiCpt = imagenet.resnet_200_final_epinet()
