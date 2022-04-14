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
"""Refactor ResNet for easier research."""
import dataclasses
import functools
from typing import Optional, Sequence

import chex
from enn import base
import haiku as hk
import jax
import jax.numpy as jnp


@dataclasses.dataclass
class ResNetConfig:
  """Configuration options for ResNet."""
  channels_per_group: Sequence[int]
  blocks_per_group: Sequence[int]
  strides_per_group: Sequence[int] = (1, 2, 2, 2)

  def __post_init__(self):
    assert len(self.channels_per_group) == self.num_groups
    assert len(self.blocks_per_group) == self.num_groups
    assert len(self.strides_per_group) == self.num_groups

  @property
  def num_groups(self) -> int:
    return len(self.channels_per_group)


class ResNet(hk.Module):
  """ResNet implementation designed for maximal clarity/simplicity.

  Now exposes the core hidden units for easier access with epinet training.
  """

  def __init__(self,
               num_classes: int,
               config: ResNetConfig,
               name: Optional[str] = None):
    super().__init__(name=name)

    # ResNet stem
    self.initial_conv = hk.Conv2D(
        output_channels=64,
        kernel_shape=7,
        stride=2,
        padding='SAME',
        with_bias=False,
        name='initial_conv',
    )

    # ResNet body
    self.blocks = []
    for group_idx in range(config.num_groups):
      for block_idx in range(config.blocks_per_group[group_idx]):
        block = ResBlockV2(
            output_channels=config.channels_per_group[group_idx],
            stride=config.strides_per_group[group_idx] if block_idx == 0 else 1,
            use_projection=block_idx == 0,
        )
        self.blocks.append(block)

    # ResNet head
    self.final_bn = hk.BatchNorm(create_scale=True, create_offset=True,
                                 decay_rate=0.9, name='final_batchnorm')
    self.final_fc = hk.Linear(num_classes, w_init=jnp.zeros, name='final_fc')

  def __call__(self,
               inputs: chex.Array,
               is_training: bool,
               test_local_stats: bool) -> base.OutputWithPrior:
    # Holds the output of hidden layers.
    extra = {}

    # Stem
    out = self.initial_conv(inputs)
    out = hk.max_pool(
        out, window_shape=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='SAME')

    # Body
    for i, block in enumerate(self.blocks):
      out = block(out, is_training, test_local_stats)
      extra[f'hidden_{i}'] = out

    # Head
    out = self.final_bn(out, is_training, test_local_stats)
    out = jax.nn.relu(out)
    pool = jnp.mean(out, axis=[1, 2])
    extra['final_out'] = pool
    return base.OutputWithPrior(train=self.final_fc(pool), extra=extra)


class ResBlockV2(hk.Module):
  """ResNet preactivation block, 1x1->3x3->1x1 plus shortcut.

  This block is designed to be maximally simplistic and readable starting point
  for variations on ResNet50. Fix bottleneck=True compared to the reference
  haiku implementation.
  """

  def __init__(self,
               output_channels: int,
               stride: int,
               use_projection: bool,
               name: Optional[str] = None):
    super().__init__(name=name)
    self._use_projection = use_projection
    width = output_channels // 4

    # Custom constructors for batchnorm and conv
    bn_ctor = functools.partial(
        hk.BatchNorm, create_scale=True, create_offset=True, decay_rate=0.9)
    conv_ctor = functools.partial(hk.Conv2D, padding='SAME', with_bias=False)

    # Blocks of batchnorm and convolutions, with shortcut.
    self.bn0 = bn_ctor(name='batchnorm_0')
    self.conv0 = conv_ctor(width, kernel_shape=1, name='conv0')
    self.bn1 = bn_ctor(name='batchnorm_1')
    self.conv1 = conv_ctor(width, stride=stride, kernel_shape=3, name='conv1')
    self.bn2 = bn_ctor(name='batchnorm_2')
    self.conv2 = conv_ctor(output_channels, kernel_shape=1, name='conv2')
    self.conv_shortcut = conv_ctor(
        output_channels, stride=stride, kernel_shape=1, name='conv_shortcut')

  def __call__(self,
               inputs: chex.Array,
               is_training: bool,
               test_local_stats: bool) -> chex.Array:
    # First layer requires special handling due to projection.
    norm = jax.nn.relu(self.bn0(inputs, is_training, test_local_stats))
    layer_1 = self.conv0(norm)
    if self._use_projection:
      shortcut = self.conv_shortcut(norm)
    else:
      shortcut = inputs

    layer_2 = self.conv1(
        jax.nn.relu(self.bn1(layer_1, is_training, test_local_stats)))
    layer_3 = self.conv2(
        jax.nn.relu(self.bn2(layer_2, is_training, test_local_stats)))
    return layer_3 + shortcut


RESNET_50 = ResNetConfig((256, 512, 1024, 2048), (3, 4, 6, 3))
RESNET_101 = ResNetConfig((256, 512, 1024, 2048), (3, 4, 23, 3))
RESNET_152 = ResNetConfig((256, 512, 1024, 2048), (3, 8, 36, 3))
RESNET_200 = ResNetConfig((256, 512, 1024, 2048), (3, 24, 36, 3))
