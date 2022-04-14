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
"""Network definitions for VGG."""

from typing import Any, Sequence, Tuple

from absl import logging
import chex
from enn import base
from enn.networks import ensembles
import haiku as hk
import jax
import jax.numpy as jnp

BatchNormIndex = Tuple[Any, bool]
_VGG_CHANNELS = (64, 64, 128, 128, 128, 256, 256, 256, 512, 512, 512)
_VGG_STRIDES = (1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1)


# PIPELINE VGG (VGG - max_pooling + batch_norm)
class VGG(hk.Module):
  """VGG Network with batchnorm and without maxpool."""

  def __init__(self,
               num_output_classes: int,
               vgg_output_channels: Sequence[int] = _VGG_CHANNELS,
               vgg_strides: Sequence[int] = _VGG_STRIDES,
               name=None,):
    super().__init__(name=name)
    logging.info('Initializing a VGG-Net.')
    self._output_channels = vgg_output_channels
    self._strides = vgg_strides
    num_channels = len(self._output_channels)
    assert len(self._strides) == num_channels
    self._kernel_shapes = [[3, 3]] * num_channels

    self._conv_modules = [
        hk.Conv2D(  # pylint: disable=g-complex-comprehension
            output_channels=self._output_channels[i],
            kernel_shape=self._kernel_shapes[i],
            stride=self._strides[i],
            name=f'conv_2d_{i}') for i in range(num_channels)
    ]
    # TODO(author2): Find a more robust way to exclude batchnorm params.
    self._bn_modules = [
        hk.BatchNorm(  # pylint: disable=g-complex-comprehension
            create_offset=True,
            create_scale=False,
            decay_rate=0.999,
            name=f'batchnorm_{i}') for i in range(num_channels)
    ]
    self._logits_module = hk.Linear(num_output_classes, name='logits')

  def __call__(self,
               inputs: chex.Array,
               is_training: bool = True,
               test_local_stats: bool = False) -> chex.Array:
    net = inputs
    for conv_layer, bn_layer in zip(self._conv_modules, self._bn_modules):
      net = conv_layer(net)
      net = bn_layer(
          net, is_training=is_training, test_local_stats=test_local_stats)
      net = jax.nn.relu(net)
    # Avg pool along axis 1 and 2
    net = jnp.mean(net, axis=[1, 2], keepdims=False, dtype=jnp.float64)
    return self._logits_module(net)


class EnsembleVGGENN(base.EpistemicNetworkWithState):
  """Ensemble of VGG Networks created using einsum ensemble."""

  def __init__(self,
               num_output_classes: int,
               num_ensemble: int = 1,
               is_training: bool = True,
               test_local_stats: bool = False):

    def net_fn(x: chex.Array) -> chex.Array:
      return VGG(num_output_classes)(
          x, is_training=is_training, test_local_stats=test_local_stats)

    transformed = hk.without_apply_rng(hk.transform_with_state(net_fn))
    enn = ensembles.EnsembleWithState(transformed, num_ensemble)
    super().__init__(enn.apply, enn.init, enn.indexer)
