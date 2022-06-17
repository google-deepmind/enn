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
"""Network definitions for LeNet5."""
from typing import Sequence

from absl import logging
import chex
from enn.networks import base as network_base
from enn.networks import ensembles
import haiku as hk
import jax

_LeNet5_CHANNELS = (6, 16, 120)


class LeNet5(hk.Module):
  """VGG Network with batchnorm and without maxpool."""

  def __init__(self,
               num_output_classes: int,
               lenet_output_channels: Sequence[int] = _LeNet5_CHANNELS,):
    super().__init__()
    logging.info('Initializing a LeNet5.')
    self._output_channels = lenet_output_channels
    num_channels = len(self._output_channels)

    self._conv_modules = [
        hk.Conv2D(  # pylint: disable=g-complex-comprehension
            output_channels=self._output_channels[i],
            kernel_shape=5,
            padding='SAME',
            name=f'conv_2d_{i}') for i in range(num_channels)
    ]
    self._mp_modules = [
        hk.MaxPool(  # pylint: disable=g-complex-comprehension
            window_shape=2, strides=2, padding='SAME',
            name=f'max_pool_{i}') for i in range(num_channels)
    ]
    self._flatten_module = hk.Flatten()
    self._linear_module = hk.Linear(84, name='linear')
    self._logits_module = hk.Linear(num_output_classes, name='logits')

  def __call__(self, inputs: chex.Array) -> chex.Array:
    net = inputs
    for conv_layer, mp_layer in zip(self._conv_modules, self._mp_modules):
      net = conv_layer(net)
      net = jax.nn.relu(net)
      net = mp_layer(net)
    net = self._flatten_module(net)
    net = self._linear_module(net)
    net = jax.nn.relu(net)
    return self._logits_module(net)


class EnsembleLeNet5ENN(network_base.EpistemicNetworkWithState):
  """Ensemble of LeNet5 Networks created using einsum ensemble."""

  def __init__(self,
               num_output_classes: int,
               num_ensemble: int = 1,):
    def net_fn(x: chex.Array) -> chex.Array:
      return LeNet5(num_output_classes)(x)
    transformed = hk.without_apply_rng(hk.transform_with_state(net_fn))
    enn = ensembles.EnsembleWithState(transformed, num_ensemble)
    super().__init__(enn.apply, enn.init, enn.indexer)
